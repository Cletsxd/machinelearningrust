use std::f64;

#[derive(Clone)]
struct Matriz {
	vector: Vec<f32>,
	rows: usize,
	columns: usize,
}

impl Matriz {
	fn create_matriz_random(rows: usize, columns: usize) -> Matriz {
		let mut vector = Vec::with_capacity(rows*columns);

		for _i in 0..(rows*columns){
			vector.push(random_number(-1,1));
		}

		Matriz {vector, rows, columns}
	}

	fn create_matriz_zeros(rows: usize, columns: usize) -> Matriz {
		let mut vector = Vec::with_capacity(rows*columns);

		for _i in 0..(rows*columns){
			vector.push(0.0);
		}

		Matriz {vector, rows, columns}
	}

	fn create_matriz_null() -> Matriz {
		let vector = Vec::with_capacity(0);
		let rows = 0;
		let columns = 0;

		Matriz {vector, rows, columns}
	}

	fn create_matriz(rows: usize, columns: usize, vector: Vec<f32>) -> Matriz {
		Matriz {vector, rows, columns}
	}

	fn t(&self) -> Matriz {
		let mut mat_r = Matriz::create_matriz_zeros(self.columns, self.rows);

		for i in 0..self.rows {
			for j in 0..self.columns {
				mat_r.vector[(j*self.rows)+i] = self.vector[(i*self.columns)+j];
			}
		}

		mat_r
	}

	fn show(&self) {
		print!("{}", "[");
		for i in 0..self.rows {
			print!("{}", "[");
			for j in 0..self.columns {
				print!("{}", self.vector[(i*self.columns)+j]);
				if j!=self.columns-1 {
					print!("{}", ", ");
				}
			}
			print!("{}", "]");
			if i!=self.rows-1 {
				println!("{}", ", ");
			}
		}
		println!("{}", "]");
	}

}

// Incluye la capa de entrada, la cual solo tiene 'output', es decir, es la matriz del set de entrenamiento o de prueba
#[derive(Clone)]
struct NeuralLayer {
	weights: Matriz,
	bias: Matriz,
	deltas: Matriz,
	output: Matriz,
	activation_fuction: Functions,
}

impl NeuralLayer {
	fn create_neural_layer(weights: Matriz, bias: Matriz, deltas: Matriz, output: &Matriz, activation_fuction: Functions) -> NeuralLayer {
		NeuralLayer{weights, bias, deltas, output: output.clone(), activation_fuction}
	}

	fn set_output(&mut self, output: &Matriz) {
		self.output = output.clone();
	}

	fn set_deltas(&mut self, deltas: &Matriz) {
		self.deltas = deltas.clone();
	}

	fn set_bias(&mut self, bias: &Matriz) {
		self.bias = bias.clone();
	}

	fn set_weights(&mut self, weights: &Matriz) {
		self.weights = weights.clone();
	}
}

struct NeuralNet {
	layers: usize,
	neural_net: Vec<NeuralLayer>,
}

impl NeuralNet {
	fn create_neural_net(layers: usize, input: Matriz, topo_nn: Vec<usize>) -> NeuralNet {
		let mut neural_net: Vec<NeuralLayer> = Vec::with_capacity(layers);

		for (index, (i, j)) in topo_nn.iter().zip(topo_nn.iter().skip(1)).enumerate() {
			if index==0 {
				let nl1 = NeuralLayer::create_neural_layer(Matriz::create_matriz_null(), Matriz::create_matriz_null(), Matriz::create_matriz_null(), &input, Functions::Sigmoidal);
				neural_net.push(nl1);
				let nlm = NeuralLayer::create_neural_layer(Matriz::create_matriz_random(*i,*j), Matriz::create_matriz_random(1,*j), Matriz::create_matriz_null(), &Matriz::create_matriz_null(), Functions::Sigmoidal);
				neural_net.push(nlm);
			}else{
				let nlm = NeuralLayer::create_neural_layer(Matriz::create_matriz_random(*i,*j), Matriz::create_matriz_random(1,*j), Matriz::create_matriz_null(), &Matriz::create_matriz_null(), Functions::Sigmoidal);
				neural_net.push(nlm);
			}
		}

		NeuralNet {layers, neural_net}
	}

	fn show(&self){
		for (i, layer) in self.neural_net.iter().enumerate() {
			if i==0 {
				print!("{}", "Input layer:\n");
				print!("{}", "Output\n");
				layer.output.show();
				print!("{}", "\n");
			}else{
				print!("Layer {}: \n", i+1);
				print!("{}", "Weights\n");
				layer.weights.show();
				print!("{}", "Bias\n");
				layer.bias.show();
				print!("{}", "\n");
			}
		}
	}

	fn show_final_output(&self) {
		match self.neural_net.last() {
			Some(i) => i.output.show(),
			None => print!("nada"),
		}
	}

	fn show_output(&self, layer: usize) {
		unimplemented!();
	}

	fn feed_forward(&mut self) {
		for (i, j) in (0..self.neural_net.len()).zip(1..self.neural_net.len()) {
			// regresión lineal
			let mut out = dot(&self.neural_net[i].output, &self.neural_net[j].weights);

			// suma con bias
			out = suma_wc(&out, &self.neural_net[j].bias);

			// función de activación
			out = self.neural_net[j].activation_fuction.active_f(&out);
			self.neural_net[j].set_output(&out);
		}
	}

	fn backpropagation(&mut self, exp_input: &Matriz, learning_rate: f32) {
		let ini = (-1 * self.neural_net.len() as i32) +1;

		//- Para cada capa:
		for (i, j) in (ini..0).zip((ini+1)..1) {
			//print!("{}, {}", i*-1, j*-1);
			//print!("\n\n > Actualizando layer {}\n", i*-1);

			//- Si es la última capa:
			if i==ini {
				//- Calcular deltas última capa:
				//> Calcular error (e2medio = output layer - exp_input)
				let error = d_e2medio(&self.neural_net[(i*-1) as usize].output, &exp_input);
				/*print!("- e2medio\n");
				error.show();*/

				//> Calcular deriv output layer
				let deriv = self.neural_net[(i*-1) as usize].activation_fuction.derived_f(&self.neural_net[(i*-1) as usize].output);
				/*print!("- output layer\n");
				self.neural_net[(i*-1) as usize].output.show();
				print!("- deriv\n");
				deriv.show();*/

				//> Calcular delta layer: error * deriv
				self.neural_net[(i*-1) as usize].set_deltas(&mult_mat(&error, &deriv));
				/*print!("- nuevas deltas\n");
				mult_mat(&error, &deriv).show();*/
			} else {
			//- Si no:
				//- Calcular deltas anteriores:
				//> Recuperación delta layer+1
				let delta = &self.neural_net[(i*-1) as usize +1].deltas.clone();
				/*print!("- deltas layer +1\n");
				delta.show();*/

				//> Recuperación weights layer+1
				let weights = &self.neural_net[(i*-1)as usize +1].weights.clone();
				/*print!("- weights layer +1\n");
				weights.show();*/

				//> Transposición weights layer+1 -> w_t layer+1
				let weightst = weights.t();
				/*print!("- weightst layer +1\n");
				weightst.show();*/

				//> Recuperación output layer
				let output = &self.neural_net[(i*-1) as usize].output.clone();
				/*print!("- output\n");
				output.show();*/

				//> Calcular doo = deriv output layer
				let doo = self.neural_net[(i*-1) as usize].activation_fuction.derived_f(&output);
				/*print!("- deriv\n");
				doo.show();*/

				//> Calcular mult_mat = dot(delta layer+1, w_t layer+1)
				let mm = dot(&delta, &weightst);
				/*print!("- dot\n");
				mm.show();*/

				//> Calcular delta layer: mult_mat * doo
				self.neural_net[(i*-1) as usize].set_deltas(&mult_mat(&mm, &doo));
				/*print!("- nuevas deltas\n");
				mult_mat(&mm, &doo).show();*/
			}
			// sigue..

			// Descenso del Gradiente

			//- Actualización de bias:
			//print!("-> actualización de bias\n");
			//> Calcular mean = mean(deltas layer)
			let mean = mean(&self.neural_net[(i*-1) as usize].deltas);
			/*print!("- mean\n");
			mean.show();*/

			//> Calcular mlr = mean * learning rate
			let mlr = mult_mat_float(&mean, learning_rate);
			/*print!("- nuevas bias\n");
			mlr.show();*/

			//> Actualizar bias layer
			let b = self.neural_net[(i*-1) as usize].bias;
			self.neural_net[(i*-1) as usize].set_bias(&resta_mat(&b, &mlr));

			//- Actualización de weights:
			//print!("-> actualización de weights\n");
			//> Recuperación output layer-1
			let out = &self.neural_net[(j*-1) as usize].output.clone();
			/*print!("- output layer -1\n");
			out.show();*/

			//> Transposición output layer-1 -> out_t layer-1
			let outt = out.t();
			/*print!("- outt layer -1\n");
			outt.show();*/

			//> Calcular do = dot(out_t layer-1, delta layer)
			let dooo = dot(&outt, &self.neural_net[(i*-1) as usize].deltas);
			/*print!("- deltas layer\n");
			self.neural_net[(i*-1) as usize].deltas.show();
			print!("- dooo (outt layer -1 @ deltas layer)\n");
			dooo.show();*/

			//> Calcular dlr = dot * learning_rate
			let dlr = mult_mat_float(&dooo, learning_rate);
			/*print!("- nuevas weights\n");
			dlr.show();*/

			//> Actualizar weights layer
			let w = self.neural_net[(i*-1) as usize].weights;
			self.neural_net[(i*-1) as usize].set_weights(&resta_mat(&w, &dlr));
		}
	}

	fn train(&mut self, exp_input: Matriz, epochs: usize, learning_rate: f32) {
		for i in 0..epochs {
			self.feed_forward();
			self.backpropagation(&exp_input, learning_rate);
		}
	}
}

fn random_number(min: i32, max: i32) -> f32 {
	assert!(min < max);

	let dif = max - min;
	let x: f32 = rand::random();

	dif as f32 * x + min as f32
}

fn dot(mat_a: &Matriz, mat_b: &Matriz) -> Matriz {
	let mat_a = mat_a.clone();
	let mat_b = mat_b.clone();

	let mut mat_r = Matriz::create_matriz_zeros(mat_a.rows, mat_b.columns);

	assert_eq!(mat_a.columns, mat_b.rows);
	
	for i in 0..mat_a.rows {
		for j in 0..mat_b.columns {
			let mut sum = 0.0;
			for k in 0..mat_a.columns {
				sum = mat_a.vector[(i*mat_a.columns)+k] * mat_b.vector[(k*mat_b.columns)+j] + sum;
			}
			mat_r.vector[(i*mat_b.columns)+j] = sum;
		}
	}

	mat_r
}

fn suma_wc(mat_a: &Matriz, mat_b: &Matriz) -> Matriz {
	let mat_a = mat_a.clone();
	let mat_b = mat_b.clone();

	assert!(mat_a.rows != mat_b.rows);
	assert_eq!(mat_a.columns, mat_b.columns);

	let mut mat_r = Matriz::create_matriz_zeros(mat_a.rows, mat_a.columns);

	for i in 0..mat_a.rows {
		for j in 0..mat_a.columns {
			mat_r.vector[(i*mat_a.columns)+j] = mat_a.vector[(i*mat_a.columns)+j] + mat_b.vector[j];
		}
	}

	mat_r
}

fn resta_mat(mat_a: &Matriz, mat_b: &Matriz) -> Matriz {
	let mat_a = mat_a.clone();
	let mat_b = mat_b.clone();

	assert_eq!(mat_a.columns, mat_b.columns);
	assert_eq!(mat_a.rows, mat_b.rows);

	let mut mat_r = Matriz::create_matriz_zeros(mat_a.rows, mat_a.columns);

	for i in 0..mat_a.rows {
		for j in 0..mat_a.columns {
			mat_r.vector[(i*mat_a.columns)+j] = mat_a.vector[(i*mat_a.columns)+j] - mat_b.vector[j];
		}
	}

	mat_r
}

fn d_e2medio(mat_a: &Matriz, mat_b: &Matriz) -> Matriz {
	let mat_a = mat_a.clone();
	let mat_b = mat_b.clone();

	assert_eq!(mat_a.rows, mat_b.rows);
	assert_eq!(mat_a.columns, mat_b.columns);

	let mut mat_r = Matriz::create_matriz_zeros(mat_a.rows, mat_a.columns);

	for i in 0..mat_a.rows {
		for j in 0..mat_a.columns {
			mat_r.vector[(i*mat_a.columns)+j] = mat_a.vector[(i*mat_a.columns)+j] - mat_b.vector[(i*mat_a.columns)+j];
		}
	}

	mat_r
}

fn mult_mat(mat_a: &Matriz, mat_b: &Matriz) -> Matriz {
	let mat_a = mat_a.clone();
	let mat_b = mat_b.clone();

	assert_eq!(mat_a.rows, mat_b.rows);
	assert_eq!(mat_a.columns, mat_b.columns);

	let mut mat_r = Matriz::create_matriz_zeros(mat_a.rows, mat_a.columns);

	for i in 0..mat_a.rows {
		for j in 0..mat_a.columns {
			mat_r.vector[(i*mat_a.columns)+j] = mat_a.vector[(i*mat_a.columns)+j] * mat_b.vector[(i*mat_a.columns)+j];
		}
	}

	mat_r
}

fn mean(mat: &Matriz) -> Matriz {
	let mat = mat.clone();

	let mut mat_r = Matriz::create_matriz_zeros(1, mat.columns);

	for i in 0..mat.rows {
		for j in 0..mat.columns {
			mat_r.vector[j] = mat.vector[(i*mat.columns)+j] + mat_r.vector[j];
		}
	}

	for j in 0..mat.columns {
		mat_r.vector[j] = mat_r.vector[j]/mat.rows as f32;
	}

	mat_r
}

fn mult_mat_float(mat: &Matriz, numberf: f32) -> Matriz {
	let mat = mat.clone();

	let mut mat_r = Matriz::create_matriz_zeros(mat.rows, mat.columns);

	for i in 0..mat.rows {
		for j in 0..mat.columns {
			mat_r.vector[(i*mat.columns)+j] = mat.vector[(i*mat.columns)+j] * numberf;
		}
	}

	mat_r
}

#[derive(Clone, Copy)]
enum Functions {
	Sigmoidal,
	Tanh,
	Relu,
}

impl Functions {
	fn active_f(&self, output: &Matriz) -> Matriz {

		let output = output.clone();

		let mut mat_r = Matriz::create_matriz_zeros(output.rows, output.columns);

		match self {
			Functions::Sigmoidal => {
				for i in 0..output.rows {
					for j in 0..output.columns {
						let val = -1_f32 * output.vector[(i*output.columns)+j];
						let exp_value = val.exp();
						mat_r.vector[(i*output.columns)+j] = 1_f32/(1_f32+exp_value as f32);
					}
		  		}
			},

			Functions::Tanh => {
				for i in 0..output.rows {
					for j in 0..output.columns {
						mat_r.vector[(i*output.columns)+j] = output.vector[(i*output.columns)+j].tanh();
					}
				}
			},

			Functions::Relu => {
				unimplemented!();
				/*
				def ReLU(x):
    				return x * (x > 0)
				*/
			},

		}

		mat_r
	}

	fn derived_f(&self, output: &Matriz) -> Matriz {
		let output = output.clone();

		let mut mat_r = Matriz::create_matriz_zeros(output.rows, output.columns);

		match self {
			Functions::Sigmoidal => {
				for i in 0..output.rows {
					for j in 0..output.columns {
						mat_r.vector[(i*output.columns)+j] = output.vector[(i*output.columns)+j] * (1_f32 - output.vector[(i*output.columns)+j]);
					}
		  		}
			},

			Functions::Tanh => {
				for i in 0..output.rows {
					for j in 0..output.columns {
						
						mat_r.vector[(i*output.columns)+j] = 1_f32 - (output.vector[(i*output.columns)+j].powf(2_f32));
					}
				}
			},

			Functions::Relu => {
				unimplemented!();
				/*
				def dReLU(x):
    				return 1. * (x > 0)
				*/
			},

		}

		mat_r
	}
}

fn main() {
	//rustup doc --std
	//rustup doc --book
	// Sobre la red
	let layers = 3;
	let mut topo_nn = Vec::with_capacity(layers);
	topo_nn = [2, 3, 1].to_vec();

	// Sobre los datos de entrada de entrenamiento
	let rows_x = 4;
	let columns_x = 2;
	let mut vec_x = Vec::with_capacity(rows_x*columns_x);
	vec_x = [0.0,0.0, 1.0,0.0, 0.0,1.0, 1.0,1.0].to_vec();
	let X = Matriz::create_matriz(rows_x, columns_x, vec_x);

	// Sobre los datos de salida esperada
	let rows_y = 4;
	let columns_y = 1;
	let mut vec_y = Vec::with_capacity(rows_y*columns_y);
	vec_y = [0.0, 1.0, 1.0, 0.0].to_vec();
	let Y = Matriz::create_matriz(rows_y, columns_y, vec_y);

	// Creación de la ANN
	let mut ann = NeuralNet::create_neural_net(layers, X, topo_nn);

	// mostrar ann
	ann.show();

	// pensar (feed-forward)
	ann.feed_forward();

	// mostrar final output
	print!("Final Output\n");
	ann.show_final_output();

	// mostrar todos los outputs
	//ann.show_outputs();

	// Training
	ann.train(Y, 2, 0.01);
	
	
	// mostrar ann
	print!("\nNeural Net after training\n");
	ann.show();

	// pensar (feed-forward)
	ann.feed_forward();

	// mostrar final output
	print!("Final Output\n");
	ann.show_final_output();

}