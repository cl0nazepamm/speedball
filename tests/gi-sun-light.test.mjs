import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as THREE from 'three';
import { collectLights } from '../js/spectral_scene.js';

test('SunLight is directional and does not inherit the previous light target', () => {
    const scene = new THREE.Scene();
    const directional = new THREE.DirectionalLight();
    directional.position.set(4, 5, 6);
    directional.target.position.set(9, 2, 1);
    const sun = THREE.SunLight ? new THREE.SunLight() : new THREE.Light();
    sun.isSunLight = true;
    sun.position.set(0, 10, 0);
    scene.add(directional, directional.target, sun);
    scene.updateMatrixWorld(true);
    const lights = collectLights(THREE, scene);
    assert.equal(lights.length, 2);
    assert.equal(lights[1][0], 0);
    const expected = new THREE.Vector3(0, -1, 0);
    assert.ok(new THREE.Vector3(...lights[1].slice(4, 7)).distanceTo(expected) < 1e-6);
});
