import assert from 'node:assert/strict';
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

console.log('webgpu cleanup smoke: ok');
