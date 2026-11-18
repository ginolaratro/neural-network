import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {makeCar, observe, predict, step} from '../public/simulation.js';
const model = JSON.parse(await readFile(new URL('../public/model.json', import.meta.url)));
test('trained network keeps the car on the road from several starting states', () => {
  for (const [offset, angle] of [[-25,-0.2], [20,0.15], [25,-0.2]]) {
    const car=makeCar(offset,angle);
    for(let i=0;i<1000 && !car.offRoad;i++) step(car,model);
    assert.equal(car.steps,1000); assert.ok(Math.abs(observe(car).offset)<10);
  }
});
test('model produces a finite probability',()=>{const [output]=predict(model,[0,0]);assert.ok(output>0 && output<1);});
