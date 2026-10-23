/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.iceberg.tornadovm;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Collection;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.apache.iceberg.io.FileIO;
import org.apache.iceberg.relocated.com.google.common.collect.Lists;

/**
 * Local paths for the files a device scan reads and writes. cuDF opens files by path, and a table
 * on an object store (S3, GCS, ADLS, through Iceberg's {@link FileIO}) has none: its files are
 * downloaded once into a local cache -- Iceberg files never change once written, so a cached copy
 * never goes stale -- and files a rewrite writes go to a local staging file that is uploaded after.
 *
 * <p>The cache is {@code iceberg.tornadovm.cacheDir} (by default under {@code java.io.tmpdir}),
 * bounded to {@code iceberg.tornadovm.cacheBytes} (64 GB by default), least recently used files
 * evicted first.
 */
final class LocalFiles {
  private static final Path CACHE =
      Paths.get(
          System.getProperty(
              "iceberg.tornadovm.cacheDir",
              System.getProperty("java.io.tmpdir") + "/iceberg-tornadovm-cache"));
  private static final long LIMIT = Long.getLong("iceberg.tornadovm.cacheBytes", 64L << 30);
  private static final int FETCHERS = Integer.getInteger("iceberg.tornadovm.fetchThreads", 16);

  // Cached files by local path, least recently used first, with their sizes.
  private static final Map<Path, Long> CACHED = new LinkedHashMap<>(16, 0.75f, true);
  private static final Map<String, Object> FETCHING = new ConcurrentHashMap<>();
  private static long cachedBytes = 0;

  private static final ExecutorService POOL =
      Executors.newFixedThreadPool(
          FETCHERS,
          runnable -> {
            Thread thread = new Thread(runnable, "tornadovm-fetch");
            thread.setDaemon(true);
            return thread;
          });

  private LocalFiles() {}

  /** Whether cuDF can open the location as it is. */
  static boolean isLocal(String location) {
    return location.startsWith("file:") || location.startsWith("/");
  }

  /** A local path for the location: itself if local, else its cached copy, fetched if need be. */
  static String path(String location, FileIO io) {
    if (location.startsWith("file:")) {
      return Paths.get(URI.create(location)).toString();
    }
    if (location.startsWith("/")) {
      return location;
    }
    Path local = CACHE.resolve(digest(location) + suffix(location));
    synchronized (FETCHING.computeIfAbsent(location, ignored -> new Object())) {
      synchronized (CACHED) {
        if (CACHED.containsKey(local) && Files.exists(local)) {
          return local.toString();
        }
      }
      try {
        Files.createDirectories(CACHE);
        Path partial = Files.createTempFile(CACHE, "fetch-", ".part");
        try (InputStream in = io.newInputFile(location).newStream()) {
          Files.copy(in, partial, StandardCopyOption.REPLACE_EXISTING);
        }
        Files.move(
            partial, local, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        remember(local, Files.size(local));
      } catch (IOException e) {
        throw new UncheckedIOException("Cannot fetch " + location, e);
      }
    }
    return local.toString();
  }

  /** Fetches the remote ones among the locations at once, ahead of their first use. */
  static void prefetch(Collection<String> locations, FileIO io) {
    List<CompletableFuture<String>> fetches = Lists.newArrayList();
    for (String location : locations) {
      if (!isLocal(location)) {
        fetches.add(CompletableFuture.supplyAsync(() -> path(location, io), POOL));
      }
    }
    fetches.forEach(CompletableFuture::join);
  }

  /** A local file to write a data file into before it is uploaded to its location. */
  static Path stagingFile() {
    try {
      Files.createDirectories(CACHE);
      return Files.createTempFile(CACHE, "write-", ".parquet");
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  /** Uploads a staged file to its location, then deletes the local copy. */
  static void upload(Path staged, String location, FileIO io) {
    try (InputStream in = Files.newInputStream(staged);
        OutputStream out = io.newOutputFile(location).create()) {
      in.transferTo(out);
    } catch (IOException e) {
      throw new UncheckedIOException("Cannot upload " + location, e);
    }
    try {
      Files.deleteIfExists(staged);
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  /** Deletes a staged file nothing was written to. */
  static void discard(Path staged) {
    try {
      Files.deleteIfExists(staged);
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  private static void remember(Path local, long bytes) {
    List<Path> evicted = Lists.newArrayList();
    synchronized (CACHED) {
      Long previous = CACHED.put(local, bytes);
      cachedBytes += bytes - (previous == null ? 0 : previous);
      java.util.Iterator<Map.Entry<Path, Long>> eldest = CACHED.entrySet().iterator();
      while (cachedBytes > LIMIT && eldest.hasNext()) {
        Map.Entry<Path, Long> entry = eldest.next();
        if (entry.getKey().equals(local)) {
          continue;
        }
        cachedBytes -= entry.getValue();
        evicted.add(entry.getKey());
        eldest.remove();
      }
    }
    for (Path path : evicted) {
      try {
        Files.deleteIfExists(path);
      } catch (IOException e) {
        // A file a reader still has open is deleted when it closes it; nothing to do here.
      }
    }
  }

  private static String suffix(String location) {
    int dot = location.lastIndexOf('.');
    return dot > location.lastIndexOf('/') ? location.substring(dot) : "";
  }

  private static String digest(String location) {
    try {
      byte[] hash =
          MessageDigest.getInstance("SHA-256").digest(location.getBytes(StandardCharsets.UTF_8));
      return HexFormat.of().formatHex(hash, 0, 16);
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException(e);
    }
  }
}
