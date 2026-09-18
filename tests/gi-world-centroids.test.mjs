import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as THREE from 'three';
import { buildSpectralScene } from '../js/spectral_scene.js';

test('density centroids follow shared geometry, parent transforms, and active instances', async () => {
    const scene = new THREE.Scene();
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, 3, 0, 0, 0, 3, 0], 3));
    geometry.computeVertexNormals();
    const material = new THREE.MeshStandardMaterial();
    const parent = new THREE.Group();
    parent.position.set(100, 40, -20);
    parent.rotation.y = Math.PI / 2;
    parent.scale.set(2, 3, 4);
    scene.add(parent);
    const instances = new THREE.InstancedMesh(geometry, material, 3);
    const matrices = [new THREE.Matrix4().makeTranslation(10, 2, 3), new THREE.Matrix4().makeTranslation(-5, 8, 0)];
    matrices.forEach((m, i) => instances.setMatrixAt(i, m));
    instances.setMatrixAt(2, new THREE.Matrix4().makeTranslation(10000, 0, 0));
    instances.count = 2;
    parent.add(instances);
    const ordinary = new THREE.Mesh(geometry, material);
    ordinary.position.set(-100, 25, 300);
    scene.add(ordinary);
    const built = await buildSpectralScene({ THREE, scene });
    assert.equal(built.error, null);
    assert.equal(built.triCount, 1, 'geometry remains one shared BLAS');
    const collect = () => {
        const points = [];
        built.forEachWorldTriangleCentroid((x, y, z) => points.push(new THREE.Vector3(x, y, z)));
        return points;
    };
    const verify = () => {
        scene.updateMatrixWorld(true);
        const expected = matrices.slice(0, instances.count).map(m => new THREE.Vector3(1, 1, 0).applyMatrix4(m).applyMatrix4(instances.matrixWorld));
        expected.push(new THREE.Vector3(1, 1, 0).applyMatrix4(ordinary.matrixWorld));
        const actual = collect();
        assert.equal(actual.length, expected.length);
        for (const point of expected) assert.ok(actual.some(p => p.distanceTo(point) < 1e-5));
    };
    verify();
    parent.position.y += 50;
    instances.count = 1;
    scene.updateMatrixWorld(true);
    built.updateTransforms();
    verify();
    geometry.dispose(); material.dispose(); instances.dispose();
});
