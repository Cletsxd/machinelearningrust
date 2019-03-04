use crate::matriz::Matriz;

pub fn random_number(min: i32, max: i32) -> f32 {
	assert!(min < max);

	let dif = max - min;
	let x: f32 = rand::random();

	dif as f32 * x + min as f32
}

pub fn dot(mat_a: &Matriz, mat_b: &Matriz) -> Matriz {
	let mat_a = mat_a.clone();
	let mat_b = mat_b.clone();

	let mut mat_r = Matriz::create_matriz_zeros(mat_a.rows(), mat_b.columns());

	assert_eq!(mat_a.columns(), mat_b.rows());
	
	for i in 0..mat_a.rows() {
		for j in 0..mat_b.columns() {
			let mut sum = 0.0;
			for k in 0..mat_a.columns() {
				sum = mat_a.vector[(i*mat_a.columns())+k] * mat_b.vector[(k*mat_b.columns())+j] + sum;
			}
			mat_r.vector[(i*mat_b.columns())+j] = sum;
		}
	}

	mat_r
}

pub fn suma_wc(mat_a: &Matriz, mat_b: &Matriz) -> Matriz {
	let mat_a = mat_a.clone();
	let mat_b = mat_b.clone();

	//assert!(mat_a.rows() != mat_b.rows());
	assert_eq!(mat_a.columns(), mat_b.columns());

	let mut mat_r = Matriz::create_matriz_zeros(mat_a.rows(), mat_a.columns());

	for i in 0..mat_a.rows() {
		for j in 0..mat_a.columns() {
			mat_r.vector[(i*mat_a.columns())+j] = mat_a.vector[(i*mat_a.columns())+j] + mat_b.vector[j];
		}
	}

	mat_r
}

pub fn resta_mat(mat_a: &Matriz, mat_b: &Matriz) -> Matriz {
	let mat_a = mat_a.clone();
	let mat_b = mat_b.clone();

	assert_eq!(mat_a.columns(), mat_b.columns());
	assert_eq!(mat_a.rows(), mat_b.rows());

	let mut mat_r = Matriz::create_matriz_zeros(mat_a.rows(), mat_a.columns());

	for i in 0..mat_a.rows() {
		for j in 0..mat_a.columns() {
			mat_r.vector[(i*mat_a.columns())+j] = mat_a.vector[(i*mat_a.columns())+j] - mat_b.vector[(i*mat_a.columns())+j];
		}
	}

	mat_r
}

/// Derivada del error cuadrático medio
pub fn d_e2medio(mat_a: &Matriz, mat_b: &Matriz) -> Matriz {
	let mat_a = mat_a.clone();
	let mat_b = mat_b.clone();

	assert_eq!(mat_a.rows(), mat_b.rows());
	assert_eq!(mat_a.columns(), mat_b.columns());

	let mut mat_r = Matriz::create_matriz_zeros(mat_a.rows(), mat_a.columns());

	for i in 0..mat_a.rows() {
		for j in 0..mat_a.columns() {
			mat_r.vector[(i*mat_a.columns())+j] = mat_a.vector[(i*mat_a.columns())+j] - mat_b.vector[(i*mat_a.columns())+j];
		}
	}

	mat_r
}

pub fn mult_mat(mat_a: &Matriz, mat_b: &Matriz) -> Matriz {
	let mat_a = mat_a.clone();
	let mat_b = mat_b.clone();

	assert_eq!(mat_a.rows(), mat_b.rows());
	assert_eq!(mat_a.columns(), mat_b.columns());

	let mut mat_r = Matriz::create_matriz_zeros(mat_a.rows(), mat_a.columns());

	for i in 0..mat_a.rows() {
		for j in 0..mat_a.columns() {
			mat_r.vector[(i*mat_a.columns())+j] = mat_a.vector[(i*mat_a.columns())+j] * mat_b.vector[(i*mat_a.columns())+j];
		}
	}

	mat_r
}

pub fn mean(mat: &Matriz) -> Matriz {
	let mat = mat.clone();

	let mut mat_r = Matriz::create_matriz_zeros(1, mat.columns());

	for i in 0..mat.rows() {
		for j in 0..mat.columns() {
			mat_r.vector[j] = mat.vector[(i*mat.columns())+j] + mat_r.vector[j];
		}
	}

	for j in 0..mat.columns() {
		mat_r.vector[j] = mat_r.vector[j]/mat.rows() as f32;
	}

	mat_r
}

pub fn mult_mat_float(mat: &Matriz, numberf: f32) -> Matriz {
	let mat = mat.clone();

	let mut mat_r = Matriz::create_matriz_zeros(mat.rows(), mat.columns());

	for i in 0..mat.rows() {
		for j in 0..mat.columns() {
			mat_r.vector[(i*mat.columns())+j] = mat.vector[(i*mat.columns())+j] * numberf;
		}
	}

	mat_r
}
