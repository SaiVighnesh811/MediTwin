import { useEffect, useRef } from 'react';
import { useReducedMotion } from 'framer-motion';

const NODE_COUNT = 180;
const CONNECT_DIST = 110;
const SPEED = 0.35;

function rand(min, max) {
  return Math.random() * (max - min) + min;
}

function initNodes(w, h) {
  return Array.from({ length: NODE_COUNT }, () => ({
    x: rand(0, w),
    y: rand(0, h),
    vx: rand(-SPEED, SPEED),
    vy: rand(-SPEED, SPEED),
    phase: rand(0, Math.PI * 2),
    radius: rand(1.5, 3),
  }));
}

export default function DigitalTwin() {
  const canvasRef = useRef(null);
  const mouse = useRef({ x: -9999, y: -9999 });
  const shouldReduce = useReducedMotion();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let nodes = [];
    let raf = null;
    let lastTime = 0;

    function resize() {
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
      nodes = initNodes(canvas.width, canvas.height);
    }

    function draw(now) {
      const dt = Math.min((now - lastTime) / 16, 3); // cap delta
      lastTime = now;

      const { width: w, height: h } = canvas;
      ctx.clearRect(0, 0, w, h);

      const t = now * 0.001;

      for (const n of nodes) {
        if (!shouldReduce) {
          // Breathing oscillation
          n.x += n.vx * dt + Math.sin(t + n.phase) * 0.08;
          n.y += n.vy * dt + Math.cos(t + n.phase * 1.3) * 0.08;

          // Mouse attraction
          const dx = mouse.current.x - n.x;
          const dy = mouse.current.y - n.y;
          const dist = Math.hypot(dx, dy);
          if (dist < 150) {
            n.x += (dx / dist) * 0.4;
            n.y += (dy / dist) * 0.4;
          }

          // Bounce
          if (n.x < 0 || n.x > w) n.vx *= -1;
          if (n.y < 0 || n.y > h) n.vy *= -1;
          n.x = Math.max(0, Math.min(w, n.x));
          n.y = Math.max(0, Math.min(h, n.y));
        }
      }

      // Draw edges
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const d = Math.hypot(dx, dy);
          if (d < CONNECT_DIST) {
            const alpha = (1 - d / CONNECT_DIST) * 0.35;
            ctx.beginPath();
            ctx.strokeStyle = `rgba(94, 234, 212, ${alpha})`;
            ctx.lineWidth = 0.8;
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.stroke();
          }
        }
      }

      // Draw nodes
      for (const n of nodes) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(94, 234, 212, 0.7)';
        ctx.fill();
      }

      raf = requestAnimationFrame(draw);
    }

    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas.parentElement);

    if (shouldReduce) {
      // Static render — draw once
      draw(0);
    } else {
      raf = requestAnimationFrame(draw);
    }

    function onMouseMove(e) {
      const rect = canvas.getBoundingClientRect();
      mouse.current.x = e.clientX - rect.left;
      mouse.current.y = e.clientY - rect.top;
    }

    canvas.addEventListener('mousemove', onMouseMove);

    return () => {
      if (raf) cancelAnimationFrame(raf);
      ro.disconnect();
      canvas.removeEventListener('mousemove', onMouseMove);
    };
  }, [shouldReduce]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      style={{ width: '100%', height: '100%', display: 'block' }}
    />
  );
}

