import { StoneProps } from '../../utils/Interfaces';

const Stone = ({ color, opacity }: StoneProps) => {
  return (
    <g>
      <filter id="stoneFilter">
        <feSpecularLighting
          result="specOut"
          specularExponent="64"
          lighting-color="white"
        >
          <fePointLight x="10" y="10" z="8" />
        </feSpecularLighting>
        <feComposite
          in="SourceGraphic"
          in2="specOut"
          operator="arithmetic"
          k1="0"
          k2="1"
          k3="1"
          k4="0"
        />
      </filter>
      <circle
        cx="15.0"
        cy="15.0"
        r="12"
        fill={color}
        fillOpacity={opacity}
        filter="url(#stoneFilter)"
      />
    </g>
  );
};

export default Stone;
