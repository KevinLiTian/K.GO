import { useEffect, useRef, useState } from 'react';

import { Board } from '../components';
import API from '../api';

const Play = () => {
  // Game ID
  const id = useRef(null);
  const [loading, setLoading] = useState(true);
  const [settings, setSettings] = useState<any>(null);

  const mode = localStorage.getItem('mode')!;
  const difficulty = localStorage.getItem('difficulty')!;
  const turn = localStorage.getItem('turn');

  const DIFFICULTY_LOOKUP: { [key: string]: string } = {
    '1': '/greedypolicy',
  };

  useEffect(() => {
    if (mode == '1') {
      API.post('/setup').then((res) => {
        id.current = res.data.id;
        setSettings({
          id: id.current,
          mode,
          api: DIFFICULTY_LOOKUP[difficulty],
          turn,
        });
        setLoading(false);
      });
    } else {
      setSettings('DUMMY');
      setLoading(false);
    }
  }, []);

  return (
    <div className="w-screen h-screen bg-[#F5F5DC]">
      <div className="w-full h-full flex justify-center items-center scale-[120%]">
        {!loading && settings && <Board settings={settings} />}
      </div>
    </div>
  );
};

export default Play;
