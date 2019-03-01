use crate::matriz::Matriz;

#[derive(Clone, Copy)]
pub enum Functions {
	Sigmoidal,
	Tanh,
	Relu,
}

impl Functions {
	pub fn active_f(&self, output: &Matriz) -> Matriz {

		let output = output.clone();

		let mut mat_r = Matriz::create_matriz_zeros(output.rows(), output.columns());

		match self {
			Functions::Sigmoidal => {
				for i in 0..output.rows() {
					for j in 0..output.columns() {
						let val = -1_f32 * output.vector[(i*output.columns())+j];
						let exp_value = val.exp();
						mat_r.vector[(i*output.columns())+j] = 1_f32/(1_f32+exp_value as f32);
					}
		  		}
			},

			Functions::Tanh => {
				for i in 0..output.rows() {
					for j in 0..output.columns() {
						mat_r.vector[(i*output.columns())+j] = output.vector[(i*output.columns())+j].tanh();
					}
				}
			},

			Functions::Relu => {
				for i in 0..output.rows() {
					for j in 0..output.columns() {
						if output.vector[(i*output.columns())+j] > 0.0 {
							mat_r.vector[(i*output.columns())+j] = output.vector[(i*output.columns())+j];
						} else {
							mat_r.vector[(i*output.columns())+j] = 0.0;
						}
					}
				}
			},

		}

		mat_r
	}

	pub fn derived_f(&self, output: &Matriz) -> Matriz {
		let output = output.clone();

		let mut mat_r = Matriz::create_matriz_zeros(output.rows(), output.columns());

		match self {
			Functions::Sigmoidal => {
				for i in 0..output.rows() {
					for j in 0..output.columns() {
						mat_r.vector[(i*output.columns())+j] = output.vector[(i*output.columns())+j] * (1_f32 - output.vector[(i*output.columns())+j]);
					}
		  		}
			},

			Functions::Tanh => {
				for i in 0..output.rows() {
					for j in 0..output.columns() {
						
						mat_r.vector[(i*output.columns())+j] = 1_f32 - (output.vector[(i*output.columns())+j].powf(2_f32));
					}
				}
			},

			Functions::Relu => {
				for i in 0..output.rows() {
					for j in 0..output.columns() {
						if output.vector[(i*output.columns())+j] > 0.0 {
							mat_r.vector[(i*output.columns())+j] = 1.0;
						} else {
							mat_r.vector[(i*output.columns())+j] = 0.0;
						}
					}
				}
			},

		}

		mat_r
	}
}
