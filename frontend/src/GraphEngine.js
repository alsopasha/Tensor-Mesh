export function desiredNodeCount(width, height) {
    const MIN_NODES = 400;
    const MAX_NODES = 3500;
    const REF_NODES = 1600;
    const REF_AREA = 1512 * 982;
    const area = Math.max(1, width * height);
    const count = Math.round(REF_NODES * (area / REF_AREA));
    return Math.max(MIN_NODES, Math.min(MAX_NODES, count));
}

function gaussian2() {
    let u = Math.random();
    let v = Math.random();
    if (u < 1e-12) u = 1e-12;
    const mag = Math.sqrt(-2 * Math.log(u));
    const theta = v * Math.PI * 2;
    return [mag * Math.cos(theta), mag * Math.sin(theta)];
}

function cloudOffset(spread, warp) {
    const sigma = spread * 0.22;
    let x;
    let y;
    let r;
    do {
        const g = gaussian2();
        x = g[0] * sigma;
        y = g[1] * sigma;
        r = Math.hypot(x, y);
    } while (r > spread * 0.72);

    const wx = x * warp.sx;
    const wy = y * warp.sy;
    return [
        wx * warp.c - wy * warp.s,
        wx * warp.s + wy * warp.c
    ];
}

function randomCloudWarp() {
    const angle = Math.random() * Math.PI * 2;
    return {
        c: Math.cos(angle),
        s: Math.sin(angle),
        sx: 0.82 + Math.random() * 0.28,
        sy: 0.82 + Math.random() * 0.28
    };
}

export class Node {
    constructor(x, y, id) {
        this.id = id;
        this.x = x;
        this.y = y;
        this.vx = 0;
        this.vy = 0;
        this.targetX = x;
        this.targetY = y;
        this.homeX = x;
        this.homeY = y;
    }
}

export class Edge {
    constructor(a, b, restLength) {
        this.a = a;
        this.b = b;
        this.restLength = restLength;
    }
}

export class GraphEngine {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.width = canvas.width;
        this.height = canvas.height;

        this.nodes = [];
        this.edges = [];

        this.mouseX = -1000;
        this.mouseY = -1000;

        this.backendLoss = undefined;
        this.backendGradNorm = undefined;
        this.backendPreds = null;

        this.ready = false;
        this.cloudWarp = randomCloudWarp();
    }

    updateFromBackend(data) {
        this.backendLoss = data.loss;
        this.backendGradNorm = data.grad_norm;
        this.backendPreds = data.preds;
    }

    clearBackend() {
        this.backendLoss = undefined;
        this.backendGradNorm = undefined;
        this.backendPreds = null;
        this.scatterToCloud();
    }

    cloudSpread(count = this.nodes.length) {
        const span = Math.min(this.width, this.height);
        return Math.max(90, Math.min(span * 0.2, 28 + Math.sqrt(count) * 4.6));
    }

    scatterToCloud() {
        const cx = this.width / 2;
        const cy = this.height / 2;
        const spread = this.cloudSpread();
        this.cloudWarp = randomCloudWarp();
        for (const n of this.nodes) {
            const [dx, dy] = cloudOffset(spread, this.cloudWarp);
            n.x = cx + dx;
            n.y = cy + dy;
            n.homeX = n.x;
            n.homeY = n.y;
            n.vx = 0;
            n.vy = 0;
        }
    }

    init(text = "alsopasha") {
        this.nodes = [];
        this.edges = [];
        this.backendPreds = null;

        const MIN_NODES = 400;
        const MAX_NODES = 3500;
        const targetCount = desiredNodeCount(this.width, this.height);
        const sampleW = 1600;
        const sampleH = 900;

        const offCanvas = document.createElement('canvas');
        offCanvas.width = sampleW;
        offCanvas.height = sampleH;
        const octx = offCanvas.getContext('2d', { willReadFrequently: true });

        octx.fillStyle = '#000';
        octx.fillRect(0, 0, sampleW, sampleH);
        octx.fillStyle = '#fff';

        const fontSize = Math.min(sampleW * 0.18, 306);
        octx.font = `400 ${fontSize}px 'Damion', cursive`;
        octx.textAlign = 'center';
        octx.textBaseline = 'middle';
        octx.fillText(text, sampleW / 2, sampleH / 2);

        const imgData = octx.getImageData(0, 0, sampleW, sampleH).data;

        const collect = (step) => {
            const found = [];
            for (let y = 0; y < sampleH; y += step) {
                for (let x = 0; x < sampleW; x += step) {
                    const idx = (y * sampleW + x) * 4;
                    if (imgData[idx] > 128) found.push({ x, y });
                }
            }
            return found;
        };

        let step = 7;
        let targets = collect(step);
        while (targets.length < targetCount && step > 1) {
            step -= 1;
            targets = collect(step);
        }

        const cap = Math.max(MIN_NODES, Math.min(MAX_NODES, targetCount));
        if (targets.length > cap) {
            const stride = targets.length / cap;
            const thinned = [];
            for (let i = 0; i < cap; i++) {
                thinned.push(targets[Math.floor(i * stride)]);
            }
            targets = thinned;
        }

        let minX = Infinity;
        let minY = Infinity;
        let maxX = -Infinity;
        let maxY = -Infinity;
        for (const t of targets) {
            if (t.x < minX) minX = t.x;
            if (t.y < minY) minY = t.y;
            if (t.x > maxX) maxX = t.x;
            if (t.y > maxY) maxY = t.y;
        }

        const bw = Math.max(1, maxX - minX);
        const bh = Math.max(1, maxY - minY);
        const pad = 0.18;
        const scale = Math.min(
            (this.width * (1 - pad * 2)) / bw,
            (this.height * (1 - pad * 2)) / bh
        );
        const ox = (this.width - bw * scale) / 2 - minX * scale;
        const oy = (this.height - bh * scale) / 2 - minY * scale;

        const cx = this.width / 2;
        const cy = this.height / 2;
        const spread = this.cloudSpread(targets.length);
        this.cloudWarp = randomCloudWarp();
        for (let i = 0; i < targets.length; i++) {
            const [dx, dy] = cloudOffset(spread, this.cloudWarp);
            const n = new Node(cx + dx, cy + dy, i);
            n.targetX = targets[i].x * scale + ox;
            n.targetY = targets[i].y * scale + oy;
            this.nodes.push(n);
        }

        const grid = {};
        const cellSize = Math.max(8, 30 * scale);
        for (const n of this.nodes) {
            const gx = Math.floor(n.targetX / cellSize);
            const gy = Math.floor(n.targetY / cellSize);
            const key = `${gx},${gy}`;
            if (!grid[key]) grid[key] = [];
            grid[key].push(n);
        }

        const connectRadius = Math.max(8, 25 * scale);
        for (const n1 of this.nodes) {
            let connections = 0;
            const gx = Math.floor(n1.targetX / cellSize);
            const gy = Math.floor(n1.targetY / cellSize);

            for (let x = gx - 1; x <= gx + 1; x++) {
                for (let y = gy - 1; y <= gy + 1; y++) {
                    const cell = grid[`${x},${y}`];
                    if (!cell) continue;

                    for (const n2 of cell) {
                        if (n1 === n2 || connections >= 3) continue;
                        const dx = n1.targetX - n2.targetX;
                        const dy = n1.targetY - n2.targetY;
                        const dist = Math.sqrt(dx * dx + dy * dy);

                        if (dist < connectRadius) {
                            let exists = false;
                            for (const e of this.edges) {
                                if ((e.a === n1 && e.b === n2) || (e.a === n2 && e.b === n1)) {
                                    exists = true;
                                    break;
                                }
                            }
                            if (!exists) {
                                this.edges.push(new Edge(n1, n2, dist));
                                connections++;
                            }
                        }
                    }
                }
            }
        }

        this.ready = true;
    }

    step() {
        if (!this.ready) return;

        const training = this.backendPreds && this.backendPreds.length === this.nodes.length;

        if (training) {
            for (let i = 0; i < this.edges.length; i++) {
                const e = this.edges[i];
                const dx = e.b.x - e.a.x;
                const dy = e.b.y - e.a.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist > 0) {
                    const diff = (dist - e.restLength) / dist;
                    const force = diff * 0.1;
                    e.a.vx += dx * force;
                    e.a.vy += dy * force;
                    e.b.vx -= dx * force;
                    e.b.vy -= dy * force;
                }
            }
        }

        const disruptedNodes = [];
        const cx = this.width / 2;
        const cy = this.height / 2;
        const cloudRadius = this.cloudSpread() * 0.55;
        for (let i = 0; i < this.nodes.length; i++) {
            const n = this.nodes[i];
            let disrupted = false;

            const dxM = n.x - this.mouseX;
            const dyM = n.y - this.mouseY;
            const distSq = dxM * dxM + dyM * dyM;
            if (distSq < (training ? 8000 : 3200)) {
                const dist = Math.sqrt(distSq);
                const f = training ? 150 / (distSq + 1) : 22 / (dist + 18);
                n.vx += (dxM / dist) * f;
                n.vy += (dyM / dist) * f;
                disrupted = true;
            }

            if (training) {
                const pred = this.backendPreds[i];
                const px = pred[0] * this.width;
                const py = pred[1] * this.height;
                n.vx += (px - n.x) * 0.15;
                n.vy += (py - n.y) * 0.15;
            } else {
                n.vx += (n.homeX - n.x) * 0.14;
                n.vy += (n.homeY - n.y) * 0.14;

                const dcx = n.x - cx;
                const dcy = n.y - cy;
                const fromCentre = Math.hypot(dcx, dcy);
                if (fromCentre > cloudRadius) {
                    const pull = (fromCentre - cloudRadius) * 0.18;
                    n.vx -= (dcx / fromCentre) * pull;
                    n.vy -= (dcy / fromCentre) * pull;
                }
            }

            n.vx *= training ? 0.78 : 0.7;
            n.vy *= training ? 0.78 : 0.7;
            n.x += n.vx;
            n.y += n.vy;

            if (disrupted && training) {
                disruptedNodes.push([i, n.x / this.width, n.y / this.height]);
            }
        }

        if (disruptedNodes.length > 0 && this.onDisrupt) {
            this.onDisrupt(disruptedNodes);
        }
    }

    render() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.width, this.height);
        if (!this.ready) return;

        const training = this.backendPreds && this.backendPreds.length === this.nodes.length;
        if (training) {
            ctx.strokeStyle = 'rgba(74, 21, 21, 0.15)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let i = 0; i < this.edges.length; i++) {
                const e = this.edges[i];
                ctx.moveTo(e.a.x, e.a.y);
                ctx.lineTo(e.b.x, e.b.y);
            }
            ctx.stroke();
        }

        ctx.fillStyle = '#4a1515';
        for (let i = 0; i < this.nodes.length; i++) {
            const n = this.nodes[i];
            ctx.beginPath();
            ctx.arc(n.x, n.y, 2.0, 0, Math.PI * 2);
            ctx.fill();
        }
    }
}
