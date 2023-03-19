import { StoneProps } from '../../utils/Interfaces';

const Stone = ({ color }: StoneProps) => {
  return (
    <g>
      <filter id="stoneFilter">
        <feSpecularLighting
          result="specOut"
          specularExponent="32"
          lightingColor="grey"
        >
          <fePointLight x="12" y="12" z="9" />
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
        cx="18.0"
        cy="18.0"
        r="16.5"
        fill={color}
        filter="url(#stoneFilter)"
      />
    </g>
  );
};

export default Stone;
