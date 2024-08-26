import './styles.css'
import { hideHeader } from './components/hideHeader.js';

document.addEventListener('DOMContentLoaded', () => {
  console.log('main.js is connected!');
  hideHeader();
});

window.addEventListener('load', () => {
  window.scrollTo(0, 0);
});

