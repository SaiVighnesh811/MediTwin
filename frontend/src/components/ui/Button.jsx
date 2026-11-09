import clsx from 'clsx';
import styles from './Button.module.css';

/**
 * Button — primary interactive element.
 * variant: 'primary' | 'secondary' | 'outline' | 'ghost'
 * size:    'sm' | 'md' | 'lg'
 * as:      HTML tag or component (default 'button')
 */
export default function Button({
  children,
  variant = 'primary',
  size = 'md',
  as: Tag = 'button',
  className,
  disabled,
  ...props
}) {
  return (
    <Tag
      className={clsx(
        styles.button,
        styles[variant],
        styles[size],
        className
      )}
      disabled={Tag === 'button' ? disabled : undefined}
      aria-disabled={Tag !== 'button' && disabled ? true : undefined}
      {...props}
    >
      {children}
    </Tag>
  );
}
