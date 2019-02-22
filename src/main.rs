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

	fn create_matriz_null() -> Matriz {
		let vector = Vec::with_capacity(0);
		let rows = 0;
		let columns = 0;

		Matriz {vector, rows, columns}
	}

	fn create_matriz(rows: usize, columns: usize, vector: Vec<f32>) -> Matriz {
		Matriz {vector, rows, columns}
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
				let nl1 = NeuralLayer::create_neural_layer(Matriz::create_matriz_null(), Matriz::create_matriz_null(), Matriz::create_matriz_null(), &input, Functions::Tanh);
				neural_net.push(nl1);
				let nlm = NeuralLayer::create_neural_layer(Matriz::create_matriz_random(*i,*j), Matriz::create_matriz_random(1,*j), Matriz::create_matriz_null(), &Matriz::create_matriz_null(), Functions::Tanh);
				neural_net.push(nlm);
			}else{
				let nlm = NeuralLayer::create_neural_layer(Matriz::create_matriz_random(*i,*j), Matriz::create_matriz_random(1,*j), Matriz::create_matriz_null(), &Matriz::create_matriz_null(), Functions::Tanh);
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

	fn think(&mut self) /*-> Matriz*/ {
		for (i, j) in (0..self.neural_net.len()).zip(1..self.neural_net.len()) {
			let mut out = dot(&self.neural_net[i].output, &self.neural_net[j].weights);
			out = self.neural_net[j].activation_fuction.active(&out);
			self.neural_net[j].set_output(&out);
		}
	}

	fn train(&self, exp_input: Matriz) {
		unimplemented!();
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

		let mut mat_r = Matriz::create_matriz(mat_a.rows, mat_b.columns, Vec::with_capacity(mat_a.rows*mat_b.columns));

		assert_eq!(mat_a.columns, mat_b.rows);
		
		for i in 0..mat_a.rows {
			for j in 0..mat_b.columns {
				let mut sum = 0.0;
				for k in 0..mat_a.columns {
					sum = mat_a.vector[(i*mat_a.columns)+k] * mat_b.vector[(k*mat_b.columns)+j] + sum;
				}
				mat_r.vector.push(sum);
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
	fn active(&self, output: &Matriz) -> Matriz {

		let output = output.clone();

		let mut mat_r = Matriz::create_matriz(output.rows, output.columns, Vec::with_capacity(output.rows*output.columns));

		let e = f64::consts::E;

		match self {
			Functions::Sigmoidal => {
				for i in 0..output.rows {
					for j in 0..output.columns {
						let exp_value = e.powf(-output.vector[(i*output.columns)+j] as f64);
						mat_r.vector.push(1_f32/(1_f32+exp_value as f32));
					}
		  		}
			},

			Functions::Tanh => {
				for i in 0..output.rows {
					for j in 0..output.columns {
						let exp_value1 = e.powf(output.vector[(i*output.columns)+j] as f64);
						let exp_value2 = e.powf(-output.vector[(i*output.columns)+j] as f64);
						mat_r.vector.push((exp_value1-exp_value2) as f32 /(exp_value1+exp_value2) as f32);
					}
		  		}
			},

			Functions::Relu => {
				unimplemented!();
			},

		}

		mat_r
	}
}

fn main() {
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
	ann.think();

	// mostrar output final
	ann.show_final_output();

	// mostrar todos los outputs
	//ann.show_outputs();

	// Training
	ann.train(Y);
}