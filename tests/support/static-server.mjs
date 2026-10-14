import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const host = '127.0.0.1';
const port = 4173;
const mountPath = '/llm-book/';
const bookRoot = fileURLToPath(new URL('../../_book/', import.meta.url));
const contentTypes = {
  '.css': 'text/css; charset=utf-8',
  '.gif': 'image/gif',
  '.html': 'text/html; charset=utf-8',
  '.ico': 'image/x-icon',
  '.jpeg': 'image/jpeg',
  '.jpg': 'image/jpeg',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.pdf': 'application/pdf',
  '.png': 'image/png',
  '.svg': 'image/svg+xml; charset=utf-8',
  '.ttf': 'font/ttf',
  '.webp': 'image/webp',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.xml': 'application/xml; charset=utf-8',
};

const server = createServer(async (request, response) => {
  try {
    const url = new URL(request.url, `http://${host}:${port}`);
    if (!url.pathname.startsWith(mountPath)) {
      response.writeHead(404).end('Not found');
      return;
    }

    const relativePath = decodeURIComponent(url.pathname.slice(mountPath.length)) || 'index.html';
    let filePath = normalize(join(bookRoot, relativePath));
    if (!filePath.startsWith(bookRoot)) {
      response.writeHead(403).end('Forbidden');
      return;
    }

    if ((await stat(filePath)).isDirectory()) {
      filePath = join(filePath, 'index.html');
    }
    const body = await readFile(filePath);
    response.writeHead(200, {
      'Content-Type': contentTypes[extname(filePath)] ?? 'application/octet-stream',
      'Content-Length': body.length,
    });
    response.end(request.method === 'HEAD' ? undefined : body);
  } catch (error) {
    const status = error.code === 'ENOENT' || error.code === 'ENOTDIR' ? 404 : 500;
    response.writeHead(status).end(status === 404 ? 'Not found' : 'Server error');
  }
});

server.listen(port, host, () => {
  console.log(`Serving _book at http://${host}:${port}${mountPath}`);
});
