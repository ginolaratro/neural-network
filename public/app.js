import {RADIUS, ROAD_WIDTH, observe, makeCar, step} from './simulation.js';
const canvas = document.querySelector('#track'), ctx = canvas.getContext('2d');
const status = document.querySelector('#status'), toggle = document.querySelector('#toggle'), message = document.querySelector('#message');
let model, car = makeCar(), paused = false, last = 0, accumulator = 0;
function draw() {
  const x = canvas.width / 2, y = canvas.height / 2;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath(); ctx.arc(x,y,RADIUS,0,Math.PI*2); ctx.strokeStyle='#45524a'; ctx.lineWidth=ROAD_WIDTH; ctx.stroke();
  for (const r of [RADIUS-ROAD_WIDTH/2, RADIUS+ROAD_WIDTH/2]) {ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.strokeStyle='#a4b5a0';ctx.lineWidth=1;ctx.stroke();}
  ctx.beginPath(); ctx.arc(x,y,RADIUS,0,Math.PI*2); ctx.strokeStyle='#d6eab0'; ctx.lineWidth=2; ctx.setLineDash([7,10]); ctx.stroke(); ctx.setLineDash([]);
  ctx.save(); ctx.translate(x+car.x,y+car.y); ctx.rotate(car.heading); ctx.fillStyle='#d3f36b'; ctx.strokeStyle='#f4ffdf'; ctx.lineWidth=1.5; ctx.beginPath();ctx.moveTo(13,0);ctx.lineTo(-9,7);ctx.lineTo(-6,0);ctx.lineTo(-9,-7);ctx.closePath();ctx.fill();ctx.stroke();ctx.restore();
  const {offset,angle}=observe(car);
  document.querySelector('#steps').textContent=car.steps.toLocaleString();
  document.querySelector('#offset').innerHTML=`${offset.toFixed(1)} <small>px</small>`;
  document.querySelector('#angle').innerHTML=`${angle.toFixed(2)} <small>rad</small>`;
}
function frame(time) {
  if (!last) last=time;
  if (!paused && model && !car.offRoad) {
    accumulator += Math.min(time-last, 100);
    while (accumulator >= 20 && !car.offRoad) { step(car,model); accumulator-=20; }
    if (car.offRoad) {status.textContent='OFF ROAD'; message.textContent='Off road — press Restart to try again'; message.hidden=false;}
  }
  last=time; draw(); requestAnimationFrame(frame);
}
toggle.addEventListener('click',()=>{paused=!paused;toggle.textContent=paused?'▶   Resume':'Ⅱ   Pause'; if(!car.offRoad) status.textContent=paused?'PAUSED':'SIMULATION RUNNING';});
document.querySelector('#reset').addEventListener('click',()=>{car=makeCar();paused=false;accumulator=0;toggle.textContent='Ⅱ   Pause';status.textContent='SIMULATION RUNNING';message.hidden=true;});
fetch('./model.json').then(response=>{if(!response.ok) throw new Error('Model could not be loaded');return response.json();}).then(data=>{model=data;document.querySelector('#loss').textContent=data.training_loss.toFixed(4);}).catch(()=>{status.textContent='MODEL UNAVAILABLE';message.textContent='Unable to load the model. Please reload the page.';message.hidden=false;});
requestAnimationFrame(frame);
