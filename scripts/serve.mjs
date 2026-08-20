import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, isAbsolute, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
const host = process.env.HOST || '127.0.0.1';
const requestedPort = Number(process.env.PORT || process.argv[2] || 8777);
const port = Number.isInteger(requestedPort) && requestedPort > 0 ? requestedPort : 8777;

const MIME = new Map([
    ['.bin', 'application/octet-stream'],
    ['.css', 'text/css; charset=utf-8'],
    ['.gltf', 'model/gltf+json; charset=utf-8'],
    ['.html', 'text/html; charset=utf-8'],
    ['.jpg', 'image/jpeg'],
    ['.jpeg', 'image/jpeg'],
    ['.js', 'text/javascript; charset=utf-8'],
    ['.mjs', 'text/javascript; charset=utf-8'],
    ['.png', 'image/png'],
]);

function resolveRequestPath(requestUrl) {
    const pathname = decodeURIComponent(new URL(requestUrl || '/', 'http://localhost').pathname);
    const candidate = resolve(root, `.${pathname === '/' ? '/index.html' : pathname}`);
    const rel = relative(root, candidate);
    if (rel === '..' || rel.startsWith(`..${process.platform === 'win32' ? '\\' : '/'}`) || isAbsolute(rel)) return null;
    return candidate;
}

const server = createServer(async (request, response) => {
    if (request.method !== 'GET' && request.method !== 'HEAD') {
        response.writeHead(405, { Allow: 'GET, HEAD' }).end('Method not allowed');
        return;
    }

    let file;
    try {
        file = resolveRequestPath(request.url);
    } catch {
        response.writeHead(400).end('Bad request');
        return;
    }
    if (!file) {
        response.writeHead(403).end('Forbidden');
        return;
    }

    try {
        const info = await stat(file);
        if (!info.isFile()) throw new Error('not a file');
        response.writeHead(200, {
            'Cache-Control': 'no-store',
            'Content-Length': info.size,
            'Content-Type': MIME.get(extname(file).toLowerCase()) || 'application/octet-stream',
        });
        if (request.method === 'HEAD') response.end();
        else createReadStream(file).pipe(response);
    } catch {
        response.writeHead(404).end('Not found');
    }
});

server.listen(port, host, () => {
    console.log(`SPEEDBALL GI demo: http://${host}:${port}/`);
});
