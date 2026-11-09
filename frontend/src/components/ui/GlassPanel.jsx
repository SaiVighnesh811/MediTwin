import clsx from 'clsx';
import styles from './GlassPanel.module.css';

/**
 * GlassPanel — frosted glass card.
 * glow:    boolean — adds teal glow shadow
 * noPad:   boolean — suppresses default inner padding (for cards that manage their own)
 * padding: string  — CSS value to override the default padding (e.g. "0.5rem 1rem")
 * as:      HTML tag (default 'div')
 */
export default function GlassPanel({
  children,
  className,
  glow = false,
  noPad = false,
  padding,
  as: Tag = 'div',
  style,
  ...props
}) {
  const paddingStyle = noPad ? { padding: 0 } : padding ? { padding } : {};
  return (
    <Tag
      className={clsx(styles.panel, glow && styles.glow, className)}
      style={{ ...paddingStyle, ...style }}
      {...props}
    >
      {children}
    </Tag>
  );
}

