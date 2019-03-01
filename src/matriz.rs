use crate::math::random_number;

#[derive(Clone)]
pub struct Matriz {
	vector: Vec<f32>,
	rows: usize,
	columns: usize,
}

impl Matriz {
	pub fn create_matriz_random(rows: usize, columns: usize) -> Matriz {
		let mut vector = Vec::with_capacity(rows*columns);

		for _i in 0..(rows*columns){
			vector.push(random_number(-1,1));
		}

		Matriz {vector, rows, columns}
	}

	pub fn create_matriz_zeros(rows: usize, columns: usize) -> Matriz {
		let mut vector = Vec::with_capacity(rows*columns);

		for _i in 0..(rows*columns){
			vector.push(0.0);
		}

		Matriz {vector, rows, columns}
	}

	pub fn create_matriz_null() -> Matriz {
		let vector = Vec::with_capacity(0);
		let rows = 0;
		let columns = 0;

		Matriz {vector, rows, columns}
	}

	pub fn create_matriz(rows: usize, columns: usize, vector: Vec<f32>) -> Matriz {
		Matriz {vector, rows, columns}
	}

	pub fn t(&self) -> Matriz {
		let mut mat_r = Matriz::create_matriz_zeros(self.columns, self.rows);

		for i in 0..self.rows {
			for j in 0..self.columns {
				mat_r.vector[(j*self.rows)+i] = self.vector[(i*self.columns)+j];
			}
		}

		mat_r
	}

	pub fn show(&self) {
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