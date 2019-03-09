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

	assert_eq!(mat_a.columns(), mat_b.rows());

	let mut mat_r = Matriz::create_matriz_zeros(mat_a.rows(), mat_b.columns());
	
	for i in 0..mat_a.rows() {
		for j in 0..mat_b.columns() {
			let mut sum = 0.0;
			for k in 0..mat_a.columns() {
				sum = mat_a[(i, k)] * mat_b[(k, j)] + sum;
			}
			mat_r[(i, j)] = sum;
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
			mat_r[(i, j)] = mat_a[(i, j)] + mat_b.vector[j];
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
			mat_r[(i, j)] = mat_a[(i, j)] - mat_b[(i, j)];
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
			mat_r[(i, j)] = mat_a[(i, j)] - mat_b[(i, j)];
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
			mat_r[(i, j)] = mat_a[(i, j)] * mat_b[(i, j)];
		}
	}

	mat_r
}

pub fn mean(mat: &Matriz) -> Matriz {
	let mat = mat.clone();

	let mut mat_r = Matriz::create_matriz_zeros(1, mat.columns());

	for i in 0..mat.rows() {
		for j in 0..mat.columns() {
			mat_r[(0, j)] = mat[(i, j)] + mat_r[(0, j)];
		}
	}

	for j in 0..mat.columns() {
		mat_r[(0, j)] = mat_r[(0 ,j)]/mat.rows() as f32;
	}

	mat_r
}

pub fn mult_mat_float(mat: &Matriz, numberf: f32) -> Matriz {
	let mat = mat.clone();

	let mut mat_r = Matriz::create_matriz_zeros(mat.rows(), mat.columns());

	for i in 0..mat.rows() {
		for j in 0..mat.columns() {
			mat_r[(i, j)] = mat[(i, j)] * numberf;
		}
	}

	mat_r
}

#[cfg(test)]
mod tests_math {
	use super::Matriz;
	use crate::math::*;

	#[test]
	fn test_random_number() {
		let mut m = Matriz::create_matriz_random(3,4);
	}

	#[test]
	fn test_dot() {
		let mut mat_a = Matriz::create_matriz(3, 4, vec![
            1.0, 0.0, 0.0, 2.0,
            0.0, 0.0, 0.0, 0.0,
            3.0, 0.0, 0.0, 4.0,
        ]);

        let mut mat_b = Matriz::create_matriz(4, 2, vec![
            1.0, 0.0,
            0.0, 0.0,
            3.0, 0.0,
            3.0, 0.0,
        ]);

		let mat_r = dot(&mat_a, &mat_b);
	}

	#[test]
	fn test_suma_wc() {
		let mut mat_a = Matriz::create_matriz(3, 4, vec![
            1.0, 0.0, 0.0, 2.0,
            0.0, 0.0, 0.0, 0.0,
            3.0, 0.0, 0.0, 4.0,
        ]);

        let mut mat_b = Matriz::create_matriz(1, 4, vec![
            1.0, 0.0, 0.0, 0.0,
        ]);

        let mat_r = suma_wc(&mat_a, &mat_b);
	}

	#[test]
	fn test_resta_mat() {
		let mut mat_a = Matriz::create_matriz(3, 4, vec![
            1.0, 0.0, 0.0, 2.0,
            0.0, 0.0, 0.0, 0.0,
            3.0, 0.0, 0.0, 4.0,
        ]);

        let mut mat_b = Matriz::create_matriz(3, 4, vec![
            1.0, 0.0, 0.0, 2.0,
            0.0, 0.0, 0.0, 0.0,
            3.0, 0.0, 0.0, 4.0,
        ]);

        let mat_r = resta_mat(&mat_a, &mat_b);
	}

	#[test]
	fn test_d_e2medio() {
		let mut mat_a = Matriz::create_matriz(3, 4, vec![
            1.0, 0.0, 0.0, 2.0,
            0.0, 0.0, 0.0, 0.0,
            3.0, 0.0, 0.0, 4.0,
        ]);

        let mut mat_b = Matriz::create_matriz(3, 4, vec![
            1.0, 0.0, 0.0, 2.0,
            0.0, 0.0, 0.0, 0.0,
            3.0, 0.0, 0.0, 4.0,
        ]);

        let mat_r = d_e2medio(&mat_a, &mat_b);
	}

	#[test]
	fn test_mult_mat() {
		let mut mat_a = Matriz::create_matriz(3, 4, vec![
            1.0, 0.0, 0.0, 2.0,
            0.0, 0.0, 0.0, 0.0,
            3.0, 0.0, 0.0, 4.0,
        ]);

        let mut mat_b = Matriz::create_matriz(3, 4, vec![
            1.0, 0.0, 0.0, 2.0,
            0.0, 0.0, 0.0, 0.0,
            3.0, 0.0, 0.0, 4.0,
        ]);

        let mat_r = mult_mat(&mat_a, &mat_b);
	}

	#[test]
	fn test_mean() {
		let mut mat_a = Matriz::create_matriz(3, 4, vec![
            1.0, 0.0, 0.0, 2.0,
            0.0, 0.0, 0.0, 0.0,
            3.0, 0.0, 0.0, 4.0,
        ]);

        let mat = mean(&mat_a);
	}

	#[test]
	fn test_mult_mat_float() {
		let mut mat_a = Matriz::create_matriz(3, 4, vec![
            1.0, 0.0, 0.0, 2.0,
            0.0, 0.0, 0.0, 0.0,
            3.0, 0.0, 0.0, 4.0,
        ]);

        let mat_r = mult_mat_float(&mat_a, 0.01);
	}
}