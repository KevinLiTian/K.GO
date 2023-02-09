export const woodTexture1 = (
  <svg>
    <filter id="woodTexture1">
      <feTurbulence type="fractalNoise" baseFrequency=".1 .001" />
      <feColorMatrix
        values="0 0 0 .11 .69
           0 0 0 .09 .38
           0 0 0 .08 .14
           0 0 0 0 1"
      />
    </filter>
    <rect x="0" y="0" width="30" height="30" filter="url(#woodTexture1)" />
  </svg>
);
