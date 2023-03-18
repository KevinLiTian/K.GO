import { useEffect, useState } from 'react';

import { Board } from '../components';

const Play = () => {
  const [loading, setLoading] = useState(true);
  const [settings, setSettings] = useState<any>(null);

  const mode = localStorage.getItem('mode')!;
  const difficulty = localStorage.getItem('difficulty')!;
  const turn = localStorage.getItem('turn');

  const DIFFICULTY_LOOKUP: { [key: string]: string } = {
    '1': '/localhost:8081',
    '2': '/localhost:8080',
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
      <div className="w-full h-full flex justify-center items-center scale-[120%]">
        {!loading && settings && <Board settings={settings} />}
      </div>
    </div>
  );
};

export default Play;
