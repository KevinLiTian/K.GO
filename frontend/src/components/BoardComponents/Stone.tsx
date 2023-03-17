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
        r="13"
        fill={color}
        filter="url(#stoneFilter)"
      />
    </g>
  );
};

export default Stone;
