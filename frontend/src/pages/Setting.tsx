import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const Setting = () => {
  const navigate = useNavigate();
  const [phase, setPhase] = useState(0);

  return (
    <div className="w-screen h-screen bg-[#F5F5DC] overflow-auto">
      <div className="w-full h-full flex justify-center items-center">
        {phase == 0 && (
          <div className="flex flex-col items-center">
            <h2 className="font-poppins text-3xl">Mode</h2>
            <div className="flex flex-col gap-5 mt-5">
              <div
                className="font-poppins text-center p-3 bg-[#cd9d6f] rounded-lg hover:scale-105 active:scale-95 transition-transform cursor-pointer"
                onClick={() => {
                  localStorage.setItem('mode', '0');
                  navigate('/play');
                }}
              >
                Human Vs. Human
              </div>
              <div
                className="font-poppins text-center p-3 bg-[#cd9d6f] rounded-lg hover:scale-105 active:scale-95 transition-transform cursor-pointer"
                onClick={() => {
                  localStorage.setItem('mode', '1');
                  setPhase((prev) => prev + 1);
                }}
              >
                Human Vs. AI
              </div>
            </div>
          </div>
        )}

        {phase == 1 && (
          <div className="flex flex-col items-center">
            <h2 className="font-poppins text-3xl">Difficulty</h2>
            <div className="flex gap-5 mt-5">
              {[1, 2, 3].map((num) => (
                <div
                  key={num}
                  className="font-poppins h-[40px] w-[40px] flex justify-center items-center bg-[#cd9d6f] rounded-lg hover:scale-105 active:scale-95 transition-transform cursor-pointer"
                  onClick={() => {
                    localStorage.setItem('difficulty', JSON.stringify(num));
                    setPhase((prev) => prev + 1);
                  }}
                >
                  {num}
                </div>
              ))}
            </div>
          </div>
        )}

        {phase == 2 && (
          <div className="flex flex-col items-center">
            <h2 className="font-poppins text-3xl">Turn</h2>
            <div className="flex gap-5 mt-5">
              <div
                className="font-poppins p-3 bg-[#cd9d6f] rounded-lg hover:scale-105 active:scale-95 transition-transform cursor-pointer"
                onClick={() => {
                  localStorage.setItem('turn', '0');
                  navigate('/play');
                }}
              >
                Black
              </div>
              <div
                className="font-poppins p-3 bg-[#cd9d6f] rounded-lg hover:scale-105 active:scale-95 transition-transform cursor-pointer"
                onClick={() => {
                  localStorage.setItem('turn', '1');
                  navigate('/play');
                }}
              >
                White
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Setting;
