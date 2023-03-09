import { useNavigate } from 'react-router-dom';

import { Board } from '../components';
import GoIcon from '../assets/go.png';
import { heroBoard } from '../utils/data';

const Home = () => {
  const navigate = useNavigate();
  const settings = {
    mode: '0',
  };

  return (
    <div className="w-screen h-screen bg-[#F5F5DC] overflow-auto">
      <div className="w-full h-[80px] flex justify-between items-center pt-5">
        <div className="flex items-center pl-8">
          <img src={GoIcon} alt="go-icon" className="w-[80px] h-[80px]" />
          <h1 className="font-poppins text-3xl">K.GO</h1>
        </div>
        <div className="flex items-center gap-5 pr-20">
          <h2
            className="font-poppins text-lg cursor-pointer"
            onClick={() => {
              navigate('/setting');
            }}
          >
            Play
          </h2>
          <h2 className="font-poppins text-lg cursor-pointer">Kifu</h2>
        </div>
      </div>

      <div className="flex flex-col lg:flex-row w-full h-[80%]">
        <div className="w-full lg:w-[50%] flex flex-col justify-center gap-3 px-12 pt-6 lg:pt-0 ">
          <div className="text-center lg:text-left">
            <h1 className="font-poppins text-4xl mb-2">Mastering</h1>
            <h1 className="font-poppins text-4xl">The Grand Game of Go</h1>
          </div>
          <p className="font-poppins text-center lg:text-left">
            The game was invented in China more than 2,500 years ago and is
            believed to be the oldest board game continuously played to the
            present day
          </p>
        </div>
        <div className="w-full lg:w-[50%] flex flex-col justify-center items-center py-10 lg:pb-0">
          <Board initial={heroBoard} />
          <p className="font-poppins mt-3">
            Lee Sedol vs AlphaGo, move 78 game 4, the God's touch
          </p>
        </div>
      </div>
    </div>
  );
};

export default Home;
