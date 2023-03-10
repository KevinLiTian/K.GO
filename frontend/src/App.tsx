import { Route, Routes } from 'react-router-dom';

import { Home, Setting, Play } from './pages';

const App = () => {
  return (
    <Routes>
      <Route path="/*" element={<Home />} />
      <Route path="/setting" element={<Setting />} />
      <Route path="/play" element={<Play />} />
    </Routes>
  );
};

export default App;
