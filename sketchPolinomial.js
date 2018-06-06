let x_vals = [];
let y_vals = [];

let a, b, c ,d;
let dragging = false;
// let slider;
// let pow = 2;
// let variables = [];

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

function setup() {
  // slider = createSlider(0, 255, 6);
  // slider.position(20, 420);

  createCanvas(400,400);
  a = tf.variable(tf.scalar(random(-1, 1)));
  b = tf.variable(tf.scalar(random(-1, 1)));
  c = tf.variable(tf.scalar(random(-1, 1)));
  d = tf.variable(tf.scalar(random(-1, 1)));
  // for (let i = 0; i <= pow; i++) {
  //   variables.push(tf.variable(tf.scalar(random(-1, 1))))
  // }
  // console.log(variables)
}

function draw() {
  if(dragging){
     let x = map(mouseX, 0, width, -1, 1);
    let y = map(mouseY, 0, height , 1, -1);
    x_vals.push(x);
    y_vals.push(y);
  }else{
    tf.tidy(() => {
    if(x_vals.length > 0){
      const ys = tf.tensor1d(y_vals);
      optimizer.minimize(() => loss(predict(x_vals),ys));
    }
  });
  }

  

  background(0);

  stroke(255);
  strokeWeight(8);
  for (let i = 0; i < x_vals.length; i++) {
    let px = map(x_vals[i], -1, 1, 0, width);
    let py = map(y_vals[i], -1, 1, height, 0);
    point(px,py);
  }

  tf.tidy(() => {
    const curveX = [];
    for (let i = -1; i < 1.01; i+=0.05) {
      curveX.push(i);
    }

    const ys = predict(curveX);
    let curveY = ys.dataSync();

    beginShape();
    noFill();
    stroke(255);
    strokeWeight(2);
    for (let i = 0; i < curveX.length; i++) {
      let x = map(curveX[i], -1 , 1, 0, width);
      let y = map(curveY[i], 1, -1, 0, height);
      vertex(x, y);
    }
    endShape();
  })

}

// function mousePressed(){
//   let x = map(mouseX, 0, width, -1, 1);
//   let y = map(mouseY, 0, height , 1, -1);
//   x_vals.push(x);
//   y_vals.push(y);
// }

function predict(x) {

  // y = a x^2 + bx + c
  const xs = tf.tensor1d(x);
  // const ys = xs.square().mul(a).add(xs.mul(b)).add(c);
  const ys = xs.pow(tf.scalar(3)).mul(a)
  .add(xs.square().mul(b))
  .add(xs.mul(c))
  .add(d);

  // let ys = xs.pow(tf.scalar(pow)).mul(variables[0]);
  // for (let i = pow-1; i > 0 ; i--) {
  //  ys.add(xs.pow(tf.scalar(i)).mul(variables[pow -i]));
  // }
  // ys.add(variables[pow]);

  return ys;
}

function loss(pred, label){
 return pred.sub(label).square().mean();
}

function mousePressed(){
  dragging = true;
}

function mouseReleased(){
  dragging = false;
}