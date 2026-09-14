import React, { useEffect, useRef } from 'react';
import { GraphEngine } from './GraphEngine';
import './index.css';

function App() {
    const canvasRef = useRef(null);
    const engineRef = useRef(null);

    const statusTextRef = useRef(null);
    const lossRef = useRef(null);
    const gradRef = useRef(null);
    const nodesRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        let engine;

        Promise.all([
            document.fonts.ready,
            document.fonts.load("400 80px 'Damion'").catch(() => {})
        ]).then(() => {
            engine = new GraphEngine(canvas);
            engineRef.current = engine;

            const BACKEND_URL = (import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000').replace(/\/$/, '');
            const WS_URL = `${BACKEND_URL.replace(/^http/, 'ws')}/ws`;

            let ws = null;
            let reconnectTimer = null;
            let closed = false;

            const currentTargets = () => {
                if (!engineRef.current || !engineRef.current.ready) return null;
                return engineRef.current.nodes.map(n => [
                    n.targetX / canvas.width,
                    n.targetY / canvas.height
                ]);
            };

            const sendInit = () => {
                const targets = currentTargets();
                if (targets && ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'init', targets }));
                }
            };

            const connect = () => {
                if (closed) return;
                ws = new WebSocket(WS_URL);

                ws.onopen = () => {
                    sendInit();
                };

                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (engineRef.current) {
                        engineRef.current.updateFromBackend(data);
                    }
                };

                ws.onclose = () => {
                    if (engineRef.current) {
                        engineRef.current.clearBackend();
                    }
                    if (closed) return;
                    reconnectTimer = setTimeout(connect, 2000);
                };
            };

            connect();

            const resize = () => {
                canvas.width = window.innerWidth;
                canvas.height = window.innerHeight;
                if (engineRef.current) {
                    engineRef.current.width = canvas.width;
                    engineRef.current.height = canvas.height;
                    engineRef.current.init("alsopasha");
                    sendInit();
                }
            };

            window.addEventListener('resize', resize);
            resize();

            engine.onDisrupt = (disruptedNodes) => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: 'disrupt',
                        data: disruptedNodes
                    }));
                }
            };

            const onMove = (e) => {
                engine.mouseX = e.clientX;
                engine.mouseY = e.clientY;
            };
            window.addEventListener('mousemove', onMove);

            const onTouch = (e) => {
                if (e.touches.length > 0) {
                    engine.mouseX = e.touches[0].clientX;
                    engine.mouseY = e.touches[0].clientY;
                }
            };
            const onTouchEnd = () => {
                engine.mouseX = -1000;
                engine.mouseY = -1000;
            };
            window.addEventListener('touchmove', onTouch);
            window.addEventListener('touchstart', onTouch);
            window.addEventListener('touchend', onTouchEnd);

            let rafId;
            let lastUpdate = 0;

            const loop = (timestamp) => {
                engine.step();
                engine.render();

                const isStable = engine.backendLoss !== undefined &&
                    engine.backendLoss < 0.0001;

                if (statusTextRef.current) {
                    let dotCount = Math.floor(timestamp / 400) % 4;
                    let dots = '.'.repeat(dotCount);
                    let paddedDots = dots.padEnd(3, ' ');

                    if (engine.backendLoss === undefined) {
                        statusTextRef.current.innerText = `Waking up backend${paddedDots}`;
                        statusTextRef.current.style.color = 'var(--text-secondary)';
                    } else if (isStable) {
                        statusTextRef.current.innerText = 'Converged';
                        statusTextRef.current.style.color = 'var(--aegean)';
                    } else {
                        statusTextRef.current.innerText = `Training${paddedDots}`;
                        statusTextRef.current.style.color = 'var(--brick)';
                    }
                }

                if (timestamp - lastUpdate > 100) {
                    lastUpdate = timestamp;
                    if (lossRef.current) {
                        lossRef.current.innerText = engine.backendLoss !== undefined ? engine.backendLoss.toFixed(5) : '0.00000';
                    }
                    if (gradRef.current) {
                        gradRef.current.innerText = engine.backendGradNorm !== undefined ? engine.backendGradNorm.toFixed(5) : '0.00000';
                    }

                    if (nodesRef.current) {
                        nodesRef.current.innerText = engine.nodes.length;
                    }
                }

                rafId = requestAnimationFrame(loop);
            };
            rafId = requestAnimationFrame(loop);

            return () => {
                window.removeEventListener('resize', resize);
                window.removeEventListener('mousemove', onMove);
                window.removeEventListener('touchmove', onTouch);
                window.removeEventListener('touchstart', onTouch);
                window.removeEventListener('touchend', onTouchEnd);
                cancelAnimationFrame(rafId);
                closed = true;
                if (reconnectTimer) clearTimeout(reconnectTimer);
                if (ws) ws.close();
            };
        });
    }, []);

    return (
        <div className="app-container">
            <canvas ref={canvasRef} className="scene" />

            <div className="hud-container">
                <div className="panel hud">
                    <div className="info-section">
                        <div className="title-row">
                            <h1 className="title">Tensor Mesh</h1>
                        </div>
                        <p className="desc" ref={statusTextRef}>Waking up backend...</p>
                    </div>

                    <div className="metrics-section">
                        <div className="metric-box">
                            <div className="metric-label">Loss</div>
                            <div className="metric-value highlight-cyan" ref={lossRef}>0.0000</div>
                        </div>
                        <div className="metric-box">
                            <div className="metric-label">Gradient</div>
                            <div className="metric-value highlight-magenta" ref={gradRef}>0.0000</div>
                        </div>
                        <div className="metric-box">
                            <div className="metric-label">Nodes</div>
                            <div className="metric-value" ref={nodesRef}>0</div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="tensormesh-footer">
                Made by Yusuf Efe "Pasha" Kivilcim
            </div>
        </div>
    );
}

export default App;
