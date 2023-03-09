export interface BoardGridProps {
  col: number;
  row: number;
}

export interface BoardEdgeProps {
  style: string;
}

export interface BoardCornerProps {
  style: string;
}

export interface StoneProps {
  color: string;
}

export interface boardProps {
  initial?: number[][];
}
