import type { SVGProps } from "react";

// Authored icon set, one consistent 1.5px stroke, inherits currentColor.
// Kept minimal to exactly what the UI uses.

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function base({ size = 18, ...props }: IconProps) {
  return {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.5,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
    ...props,
  };
}

/** Brand mark: a shield with a scanning eye. */
export function ShieldEye(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M12 3l7 3v5.5c0 4.3-2.9 7.6-7 9-4.1-1.4-7-4.7-7-9V6l7-3z" />
      <circle cx="12" cy="11" r="2.1" />
      <path d="M8.2 11c1-1.7 2.4-2.6 3.8-2.6s2.8.9 3.8 2.6c-1 1.7-2.4 2.6-3.8 2.6S9.2 12.7 8.2 11z" />
    </svg>
  );
}

export function AlertTriangle(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M10.3 4.3 2.5 18a1.9 1.9 0 0 0 1.7 2.8h15.6A1.9 1.9 0 0 0 21.5 18L13.7 4.3a1.9 1.9 0 0 0-3.4 0z" />
      <path d="M12 9.5v4.5" />
      <path d="M12 17.5h.01" />
    </svg>
  );
}

export function Activity(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M3 12h3.5l2.2-6 4.6 12 2.2-6H21" />
    </svg>
  );
}

export function Camera(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M4 8.5A1.5 1.5 0 0 1 5.5 7h1.7l1-1.6a1 1 0 0 1 .84-.46h4.02a1 1 0 0 1 .84.46l1 1.6h1.7A1.5 1.5 0 0 1 20 8.5v8A1.5 1.5 0 0 1 18.5 18h-13A1.5 1.5 0 0 1 4 16.5z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

export function Bell(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M6.5 10a5.5 5.5 0 0 1 11 0c0 4 1.5 5.5 1.5 5.5H5s1.5-1.5 1.5-5.5z" />
      <path d="M10 19a2 2 0 0 0 4 0" />
    </svg>
  );
}

export function Crosshair(props: IconProps) {
  return (
    <svg {...base(props)}>
      <circle cx="12" cy="12" r="7.5" />
      <path d="M12 2.5v4M12 17.5v4M2.5 12h4M17.5 12h4" />
    </svg>
  );
}

export function Cpu(props: IconProps) {
  return (
    <svg {...base(props)}>
      <rect x="7" y="7" width="10" height="10" rx="2" />
      <path d="M9.5 2.5v2.5M14.5 2.5v2.5M9.5 19v2.5M14.5 19v2.5M2.5 9.5H5M2.5 14.5H5M19 9.5h2.5M19 14.5h2.5" />
    </svg>
  );
}

export function ArrowRight(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M5 12h14" />
      <path d="M13 6l6 6-6 6" />
    </svg>
  );
}

export function ArrowLeft(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M19 12H5" />
      <path d="M11 6l-6 6 6 6" />
    </svg>
  );
}

export function Play(props: IconProps) {
  return (
    <svg {...base({ ...props })} fill="currentColor" stroke="none">
      <path d="M8 5.5v13l11-6.5-11-6.5z" />
    </svg>
  );
}

export function Square(props: IconProps) {
  return (
    <svg {...base({ ...props })} fill="currentColor" stroke="none">
      <rect x="6.5" y="6.5" width="11" height="11" rx="2" />
    </svg>
  );
}
