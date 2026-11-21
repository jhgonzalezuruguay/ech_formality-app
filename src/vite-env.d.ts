/// <reference types="vite/client" />

declare module 'react-katex' {
  import type { ComponentType, CSSProperties, ReactNode } from 'react';

  export interface KatexProps {
    math?: string;
    children?: ReactNode;
    errorColor?: string;
    renderError?: (error: Error) => ReactNode;
    settings?: Record<string, unknown>;
    className?: string;
    style?: CSSProperties;
  }

  export const BlockMath: ComponentType<KatexProps>;
  export const InlineMath: ComponentType<KatexProps>;
}
