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
						let val = -1_f32 * output[(i, j)];
						let exp_value = val.exp();
						mat_r[(i, j)] = 1_f32/(1_f32+exp_value as f32);
					}
				}
			},

			Functions::Tanh => {
				for i in 0..output.rows() {
					for j in 0..output.columns() {
						mat_r[(i, j)] = output[(i, j)].tanh();
					}
				}
			},

			Functions::Relu => {
				for i in 0..output.rows() {
					for j in 0..output.columns() {
						if output[(i, j)] > 0.0 {
							mat_r[(i, j)] = output[(i, j)];
						} else {
							mat_r[(i, j)] = 0.0;
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
						mat_r[(i, j)] = output[(i, j)] * (1_f32 - output[(i, j)]);
					}
				}
			},

			Functions::Tanh => {
				for i in 0..output.rows() {
					for j in 0..output.columns() {
						
						mat_r[(i, j)] = 1_f32 - (output[(i, j)].powf(2_f32));
					}
				}
			},

			Functions::Relu => {
				for i in 0..output.rows() {
					for j in 0..output.columns() {
						if output[(i, j)] > 0.0 {
							mat_r[(i, j)] = 1.0;
						} else {
							mat_r[(i, j)] = 0.0;
						}
					}
				}
			},

		}

		mat_r
	}
}

#[cfg(test)]
mod tests_functions {
	use super::Functions;
	use super::Matriz;

	#[test]
	fn test_act_sigmoidal() {
		let mut m = Matriz::create_matriz(3, 4, vec![
        	1.0, 0.0, 0.0, 2.0,
        	0.0, 0.0, 0.0, 0.0,
        	3.0, 0.0, 0.0, 4.0,
    	]);

        let f = Functions::Sigmoidal;
        f.active_f(&m);
	}

	#[test]
	fn test_act_tanh() {
		let mut m = Matriz::create_matriz(3, 4, vec![
        	1.0, 0.0, 0.0, 2.0,
        	0.0, 0.0, 0.0, 0.0,
        	3.0, 0.0, 0.0, 4.0,
    	]);

        let f = Functions::Tanh;
        f.active_f(&m);
	}

	#[test]
	fn test_act_relu() {
		let mut m = Matriz::create_matriz(3, 4, vec![
        	1.0, 0.0, 0.0, 2.0,
        	0.0, 0.0, 0.0, 0.0,
        	3.0, 0.0, 0.0, 4.0,
    	]);

		let f = Functions::Relu;
        f.active_f(&m);
	}

	#[test]
	fn test_der_sigmoidal() {
		let mut m = Matriz::create_matriz(3, 4, vec![
        	1.0, 0.0, 0.0, 2.0,
        	0.0, 0.0, 0.0, 0.0,
        	3.0, 0.0, 0.0, 4.0,
    	]);

		let f = Functions::Sigmoidal;
        f.derived_f(&m);
	}

	#[test]
	fn test_der_tanh() {
		let mut m = Matriz::create_matriz(3, 4, vec![
        	1.0, 0.0, 0.0, 2.0,
        	0.0, 0.0, 0.0, 0.0,
        	3.0, 0.0, 0.0, 4.0,
    	]);

		let f = Functions::Tanh;
        f.derived_f(&m);
	}

	#[test]
	fn test_der_relu() {
		let mut m = Matriz::create_matriz(3, 4, vec![
        	1.0, 0.0, 0.0, 2.0,
        	0.0, 0.0, 0.0, 0.0,
        	3.0, 0.0, 0.0, 4.0,
    	]);

		let f = Functions::Relu;
        f.derived_f(&m);
	}

}