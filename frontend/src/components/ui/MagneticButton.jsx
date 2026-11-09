import { useRef } from 'react';
import { motion, useMotionValue, useSpring, useReducedMotion } from 'framer-motion';
import Button from './Button';

const SPRING = { stiffness: 200, damping: 20, mass: 0.5 };
const canHover =
  typeof window !== 'undefined' &&
  window.matchMedia('(hover: hover) and (pointer: fine)').matches;

/**
 * MagneticButton — wraps Button with a magnetic spring hover effect.
 * On touch/non-hover devices it renders a plain Button.
 */
export default function MagneticButton({ children, strength = 0.35, ...props }) {
  const ref = useRef(null);
  const shouldReduce = useReducedMotion();

  const rawX = useMotionValue(0);
  const rawY = useMotionValue(0);
  const x = useSpring(rawX, SPRING);
  const y = useSpring(rawY, SPRING);

  if (!canHover || shouldReduce) {
    return <Button {...props}>{children}</Button>;
  }

  function handleMouseMove(e) {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    rawX.set((e.clientX - cx) * strength);
    rawY.set((e.clientY - cy) * strength);
  }

  function handleMouseLeave() {
    rawX.set(0);
    rawY.set(0);
  }

  return (
    <motion.div
      ref={ref}
      style={{ x, y, display: 'inline-flex' }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      <Button {...props}>{children}</Button>
    </motion.div>
  );
}

