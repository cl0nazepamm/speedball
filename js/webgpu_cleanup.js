// WebGPU resource teardown shared by the DDGI field and spectral tracer.
//
// Three r185's BufferAttribute.dispose() emits a dispose event, but standalone
// StorageBufferAttributes owned only by compute bindings are not registered with
// the geometry disposal path. Renderer.info.memoryMap also retains attributes
// strongly, so dropping the compute graph is not enough: explicitly deleting the
// attribute from the renderer's attribute manager is required to destroy the
// GPUBuffer and release the renderer bookkeeping entry.

export function disposeComputeNodes(owner, keys) {
    if (!owner) return;
    for (const key of keys) owner[key]?.dispose?.();
}

export function disposeStorageAttribute(renderer, attribute) {
    if (!attribute) return;

    // Keep the public notification for compatibility with Three revisions that
    // wire BufferAttribute disposal into the renderer themselves.
    attribute.dispose?.();

    // r185 does not consume that event for standalone compute storage. The
    // manager delete is idempotent, so this is also safe when a newer renderer
    // has already removed the attribute in response to dispose().
    const attributes = renderer?._attributes;
    if (attributes?.has?.(attribute)) attributes.delete(attribute);
}

export function disposeStorageAttributes(renderer, owner, keys) {
    if (!owner) return;
    for (const key of keys) disposeStorageAttribute(renderer, owner[key]);
}
