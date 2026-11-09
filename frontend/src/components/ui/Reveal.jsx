import { motion, useReducedMotion } from 'framer-motion';

const variants = {
  hidden: (direction) => {
    const map = {
      up: { opacity: 0, y: 32 },
      down: { opacity: 0, y: -32 },
      left: { opacity: 0, x: 32 },
      right: { opacity: 0, x: -32 },
    };
    return map[direction] ?? { opacity: 0, y: 24 };
  },
  visible: { opacity: 1, x: 0, y: 0 },
};

/**
 * Reveal — wraps children in a Framer Motion enter animation.
 * Respects prefers-reduced-motion: renders children directly without animation.
 */
export default function Reveal({
  children,
  delay = 0,
  direction = 'up',
  duration = 0.55,
  className,
  ...props
}) {
  const shouldReduce = useReducedMotion();

  if (shouldReduce) {
    return (
      <div className={className} {...props}>
        {children}
      </div>
    );
  }

  return (
    <motion.div
      className={className}
      custom={direction}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: '-60px' }}
      transition={{ duration, delay, ease: [0.22, 0.03, 0.26, 1] }}
      variants={variants}
      {...props}
    >
      {children}
    </motion.div>
  );
}

