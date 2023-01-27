import { StoneProps } from '../../utils/Interfaces';

const Stone = ({ color, opacity }: StoneProps) => {
  return (
    <g>
      <circle cx="15.0" cy="15.0" r="12" fill={color} fillOpacity={opacity} />
    </g>
  );
};

export default Stone;
