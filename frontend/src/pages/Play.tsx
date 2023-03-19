import { useEffect, useState } from 'react';

import { Board } from '../components';
import GoIcon from '../assets/go.png';
import { useNavigate } from 'react-router-dom';

const Play = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [settings, setSettings] = useState<any>(null);

  const mode = localStorage.getItem('mode')!;
  const difficulty = localStorage.getItem('difficulty')!;
  const turn = localStorage.getItem('turn');

  const DIFFICULTY_LOOKUP: { [key: string]: string } = {
    '1': '/localhost:8080',
    '2': '/localhost:8081',
    '3': '/localhost:8082',
  };

  useEffect(() => {
    if (mode == '1') {
      setSettings({
        mode,
        api: DIFFICULTY_LOOKUP[difficulty],
        turn,
      });
      setLoading(false);
    } else {
      setSettings('DUMMY');
      setLoading(false);
    }
  }, []);

  return (
    <div className="w-screen h-screen bg-[#F5F5DC]">
      <div
        className="flex items-center pl-8 cursor-pointer"
        onClick={() => navigate('/')}
      >
        <img src={GoIcon} alt="go-icon" className="w-[80px] h-[80px]" />
        <h1 className="font-poppins text-3xl">K.GO</h1>
      </div>
      <div className="w-full h-4/5 flex justify-center items-center">
        {!loading && settings && <Board settings={settings} />}
      </div>
    </div>
  );
};

export default Play;
