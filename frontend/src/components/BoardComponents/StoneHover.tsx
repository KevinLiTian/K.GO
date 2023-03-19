import { StoneProps } from '../../utils/Interfaces';

const StoneHover = ({ color }: StoneProps) => {
  return (
    <g>
      <circle cx="18" cy="18" r="16.5" fill={color} opacity={0.5} />
    </g>
  );
};

export default StoneHover;
