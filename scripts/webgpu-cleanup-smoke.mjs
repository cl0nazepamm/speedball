import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import {
    disposeComputeNodes,
    disposeStorageAttribute,
    disposeStorageAttributes,
} from '../js/webgpu_cleanup.js';

const events = [];
const resident = new Set();
const renderer = {
    _attributes: {
        has: (attribute) => resident.has(attribute),
        delete(attribute) {
            events.push(`delete:${attribute.name}`);
            resident.delete(attribute);
        },
    },
};

function attribute(name) {
    const value = {
        name,
        dispose() { events.push(`dispose:${name}`); },
    };
    resident.add(value);
    return value;
}

const gpu = {
    traceKernel: { dispose() { events.push('kernel:trace'); } },
    uploadKernel: { dispose() { events.push('kernel:upload'); } },
};
const buffers = {
    bvhNodes: attribute('bvhNodes'),
    materials: attribute('materials'),
};

disposeComputeNodes(gpu, ['traceKernel', 'uploadKernel', 'missingKernel']);
disposeStorageAttributes(renderer, buffers, ['bvhNodes', 'materials', 'missingBuffer']);

assert.deepEqual(events, [
    'kernel:trace',
    'kernel:upload',
    'dispose:bvhNodes',
    'delete:bvhNodes',
    'dispose:materials',
    'delete:materials',
]);
assert.equal(resident.size, 0);

events.length = 0;
disposeStorageAttribute(renderer, buffers.bvhNodes);
assert.deepEqual(events, ['dispose:bvhNodes']);

// The one-call installer owns only the lights factory it installed. Teardown is
// idempotent, cannot clobber a later host factory, and recompiles every lit
// material after the probe resources have been retired.
const installSource = await readFile(new URL('../js/install.js', import.meta.url), 'utf8');
assert.match(installSource, /let installedCreateNode = null;/);
assert.match(installSource, /renderer\.lighting\.createNode = installedCreateNode;/);
assert.match(installSource,
    /catch \(error\) \{[\s\S]*renderer\.lighting\?\.createNode === installedCreateNode[\s\S]*renderer\.lighting\.createNode = prevCreateNode;[\s\S]*throw error;/,
    'a failed field install rolls back only its own lights factory');
assert.match(installSource,
    /try \{\s*if \(prepareMaterials\) prepareMaterialsForGI\(scene\);\s*gi = createProbeField/,
    'optional material preparation is covered by the same factory rollback');
assert.match(installSource, /cam\.getWorldPosition\(_worldPos\);[\s\S]*cam\.getWorldQuaternion\(_worldQuat\);/,
    'camera interaction follows parented world transforms');
const teardown = installSource.slice(installSource.lastIndexOf('/** Full teardown:'));
assert.match(teardown, /if \(disposed\) return;/, 'installer disposal is idempotent');
assert.match(teardown,
    /renderer\.lighting\?\.createNode === installedCreateNode[\s\S]*renderer\.lighting\.createNode = prevCreateNode;/,
    'teardown preserves a factory installed later by the host');
assert.match(teardown, /try \{\s*gi\.dispose\(\);\s*\} finally \{[\s\S]*markMaterialsDirty\(\);/,
    'materials are invalidated after the field is disposed');
assert.doesNotMatch(installSource, /if \(m\.visible === false\) return;/,
    'hidden lit materials are also invalidated before they can be shown again');
assert.match(installSource, /setNirSensing\(on\) \{\s*if \(disposed\) return;/,
    'a stale install handle cannot mutate shared direct-light sensing state');
assert.match(installSource, /setNirGain\(gain\) \{\s*if \(disposed\) return;/,
    'a stale install handle cannot mutate shared direct-light gain state');

const probesSource = await readFile(new URL('../js/gi_probes.js', import.meta.url), 'utf8');
assert.match(probesSource, /let _activeProbeFieldOwner = null;/);
assert.match(probesSource, /if \(_activeProbeFieldOwner !== null\) \{\s*throw new Error\('createProbeField: only one active field/,
    'a second live field fails explicitly instead of sharing global atlas state');
assert.match(probesSource, /if \(_activeProbeFieldOwner === fieldOwner\) _activeProbeFieldOwner = null;/,
    'disposing the owner permits a later field');
assert.match(probesSource, /if \(_node === node\) _node = null;/,
    'a later field receives a fresh probe node isolated from disposed raw handles');

console.log('webgpu cleanup smoke: ok');
