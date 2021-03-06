use crate::math::random_number;
use std::ops::{Index, IndexMut};

#[derive(Clone)]
pub struct Matriz {
    pub vector: Vec<f32>,
    columns: usize,
}

impl Matriz {
    pub fn create_matriz_random(rows: usize, columns: usize) -> Matriz {
        let mut vector = Vec::with_capacity(rows * columns);

        for _i in 0..(rows * columns) {
            vector.push(random_number(-1, 1));
        }

        Matriz { vector, columns }
    }

    pub fn create_matriz_zeros(rows: usize, columns: usize) -> Matriz {
        let mut vector = Vec::with_capacity(rows * columns);

        for _i in 0..(rows * columns) {
            vector.push(0.0);
        }

        Matriz { vector, columns }
    }

    pub fn create_matriz_null() -> Matriz {
        let vector = Vec::with_capacity(0);
        let rows = 0;
        let columns = 0;

        Matriz { vector, columns }
    }

    pub fn create_matriz(rows: usize, columns: usize, vector: Vec<f32>) -> Matriz {
        Matriz { vector, columns }
    }

    pub fn t(&self) -> Matriz {
        let mut mat_r = Matriz::create_matriz_zeros(self.columns, self.rows());

        for i in 0..self.rows() {
            for j in 0..self.columns {
                mat_r.vector[(j * self.rows()) + i] = self[(i, j)];
            }
        }

        mat_r
    }

    pub fn show(&self) {
        print!("{}", "[");

        for i in 0..self.rows() {
            print!("{}", "[");

            for j in 0..self.columns {
                print!("{}", self[(i, j)]);

                if j != self.columns - 1 {
                    print!("{}", ", ");
                }
            }

            print!("{}", "]");

            if i != self.rows() - 1 {
                println!("{}", ", ");
            }
        }

        println!("{}", "]");
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn rows(&self) -> usize {
        self.vector.len() / self.columns
    }
}

impl Index<(usize, usize)> for Matriz {
    type Output = f32;

    fn index(&self, (i, j): (usize, usize)) -> &f32 {
        assert!(i < self.rows());
        assert!(j < self.columns);

        &self.vector[self.columns * i + j]
    }
}

impl IndexMut<(usize, usize)> for Matriz {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f32 {
        &mut self.vector[self.columns * i + j]
    }
}

#[cfg(test)]
mod tests_matriz {
    use super::Matriz;

    #[test]
    fn test_matrix_index() {
        let mut m = Matriz::create_matriz(
            3,
            4,
            vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0],
        );

        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 3)], 2.0);
        assert_eq!(m[(2, 0)], 3.0);
        assert_eq!(m[(2, 3)], 4.0);

        m[(1, 1)] = 7.0;

        assert_eq!(m[(1, 1)], 7.0);
    }

    #[test]
    fn test_show() {
        let mut m = Matriz::create_matriz(
            3,
            4,
            vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0],
        );

        m.show();
    }

    #[test]
    fn test_t() {
        let mut m = Matriz::create_matriz(
            3,
            4,
            vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0],
        );

        let mt = m.t();
    }
}
