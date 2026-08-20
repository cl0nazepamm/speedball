// Cross-rebuild BLAS cache smoke: a structural rebuild must pay only for the
// geometries that actually changed. Source-text checks (repo smoke idiom: no
// three/three-mesh-bvh dependency in node) that the cache is really wired:
// keyed by attribute identity + version (a swapped-in attribute restarts its
// version counter, so identity is load-bearing), LRU-touched on hit, budgeted
// by triangles on miss, never evicting the entry the current build just made,
// and — critically — handing each build a CLONE of the cached core so a newer
// build's pool offsets can never corrupt an older build still draining async
// deform slices.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const scene = await readFile(new URL('../js/spectral_scene.js', import.meta.url), 'utf8');
const probes = await readFile(new URL('../js/gi_probes.js', import.meta.url), 'utf8');

// Cache factory + bounded budget, exported for hosts.
assert.match(scene, /export function createBlasCache\(\{ maxTriangles = 2_000_000 \} = \{\}\)/);
assert.match(scene, /map: new Map\(\), maxTriangles, triangles: 0, hits: 0, misses: 0/);

// Key: geometry uuid + attribute identity.version for index/pos/normal + the
// per-tri uber mapping. Identity must come from a WeakMap, not the version.
assert.match(scene, /const attrIds = new WeakMap\(\)/);
assert.match(scene, /\$\{geom\.uuid\}:\$\{attrIdentity\(index\)\}/);
assert.match(scene, /\$\{attrIdentity\(pos\)\}\.\$\{pos\.version\}/);
assert.match(scene, /\$\{attrIdentity\(normalAttr\)\}\.\$\{normalAttr \? normalAttr\.version : -1\}/);

// Hit path: LRU touch (delete + re-insert) and NO rebuild.
assert.match(scene, /blasCache\.hits\+\+;\s*\n\s*blasCache\.map\.delete\(key\);[^\n]*\n\s*blasCache\.map\.set\(key, core\);/);

// Miss path: insert, account triangles, evict oldest-first without ever
// evicting the entry this very build produced, and never below one entry.
assert.match(scene, /blasCache\.misses\+\+;/);
assert.match(scene, /blasCache\.triangles \+= core\.triCount;/);
assert.match(scene, /for \(const \[oldKey, old\] of blasCache\.map\) \{\s*\n\s*if \(blasCache\.triangles <= blasCache\.maxTriangles \|\| blasCache\.map\.size <= 1\) break;\s*\n\s*if \(old === core\) continue;/);
assert.match(scene, /blasCache\.triangles -= old\.triCount;/);

// The build must stamp per-build state on a CLONE of the cached core, and the
// clone must be created BEFORE the first src* stamp.
const cloneAt = scene.indexOf('const blas = Object.assign({}, core);');
const stampAt = scene.indexOf('blas.srcGeom = geom;');
assert.ok(cloneAt > 0 && stampAt > cloneAt, 'clone must precede per-build stamps');

// buildSpectralScene accepts the cache; nothing is cached when none is passed.
assert.match(scene, /maxTriangles = 4_000_000,[\s\S]*?blasCache = null,[\s\S]*?mapsArena = null,[\s\S]*?\} = \{\}/);

// gi_probes owns one cache per field: created with the lazy builder import,
// passed into every rebuild, dropped on dispose, surfaced in _debugState.
assert.match(probes, /_createBlasCache = mod\.createBlasCache \|\| null;/);
assert.match(probes, /if \(!blasCache && _createBlasCache\) blasCache = _createBlasCache\(\);/);
assert.match(probes, /buildSpectralScene\(\{[\s\S]*?THREE,[\s\S]*?scene,[\s\S]*?maxTriangles: MAX_TRIANGLES,[\s\S]*?blasCache,[\s\S]*?mapsArena,[\s\S]*?\}\)/);
assert.match(probes, /cachedBuilt = null;\s*\n\s*blasCache = null;/);
assert.match(probes, /hits: blasCache\.hits, misses: blasCache\.misses/);

console.log('gi-blas-cache-smoke: OK');
