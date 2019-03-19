use crate::functions::Functions;
use crate::math::*;
use crate::matriz::Matriz;

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
    fn create_neural_layer(
        weights: Matriz,
        bias: Matriz,
        deltas: Matriz,
        output: &Matriz,
        activation_fuction: Functions,
    ) -> NeuralLayer {
        NeuralLayer {
            weights,
            bias,
            deltas,
            output: output.clone(),
            activation_fuction,
        }
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

// Estructura de una Red Neuronal
pub struct NeuralNet {
    layers: usize,
    neural_net: Vec<NeuralLayer>,
}

impl NeuralNet {
    pub fn create_neural_net(layers: usize, input: Matriz, topo_nn: Vec<usize>) -> NeuralNet {
        let mut neural_net: Vec<NeuralLayer> = Vec::with_capacity(layers);

        for (index, (i, j)) in topo_nn.iter().zip(topo_nn.iter().skip(1)).enumerate() {
            if index == 0 {
                let nl1 = NeuralLayer::create_neural_layer(
                    Matriz::create_matriz_null(),
                    Matriz::create_matriz_null(),
                    Matriz::create_matriz_null(),
                    &input,
                    Functions::Tanh,
                );
                neural_net.push(nl1);
                let nlm = NeuralLayer::create_neural_layer(
                    Matriz::create_matriz_random(*i, *j),
                    Matriz::create_matriz_random(1, *j),
                    Matriz::create_matriz_null(),
                    &Matriz::create_matriz_null(),
                    Functions::Tanh,
                );
                neural_net.push(nlm);
            } else {
                let nlm = NeuralLayer::create_neural_layer(
                    Matriz::create_matriz_random(*i, *j),
                    Matriz::create_matriz_random(1, *j),
                    Matriz::create_matriz_null(),
                    &Matriz::create_matriz_null(),
                    Functions::Tanh,
                );
                neural_net.push(nlm);
            }
        }

        NeuralNet { layers, neural_net }
    }

    // Muestra los datos y estructura de la Red Neuronal
    pub fn show(&self) {
        for (i, layer) in self.neural_net.iter().enumerate() {
            if i == 0 {
                print!("{}", "Input layer:\n");
                print!("{}", "Output\n");
                layer.output.show();
                print!("{}", "\n");
            } else {
                print!("Layer {}: \n", i + 1);
                print!("{}", "Weights\n");
                layer.weights.show();
                print!("{}", "Bias\n");
                layer.bias.show();
                print!("{}", "\n");
            }
        }
    }

    // Muestra la salida de la red
    pub fn show_final_output(&self) {
        match self.neural_net.last() {
            Some(i) => i.output.show(),
            None => print!("nada"),
        }
    }

    pub fn show_output(&self, layer: usize) {
        unimplemented!();
    }

    // Pasa por la red neuronal hacia adelante
    pub fn feed_forward(&mut self) {
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

    // Pasa por la red neuronal hacia adelante con una entrada de prueba
    pub fn feed_forward_wi(&mut self, input: &Matriz) {
        let input = input.clone();

        for i in 1..self.neural_net.len() {
            let mut out;

            // regresión lineal
            if i==1 {
                out = dot(&input, &self.neural_net[i].weights);
            } else {
                out = dot(&self.neural_net[i - 1].output, &self.neural_net[i].weights);
            }

            // suma con bias
            out = suma_wc(&out, &self.neural_net[i].bias);

            // función de activación
            out = self.neural_net[i].activation_fuction.active_f(&out);

            self.neural_net[i].set_output(&out);
        }
    }

    // Pasa por la red hacia atrás
    pub fn backpropagation(&mut self, exp_output: &Matriz, learning_rate: f32) {
        let ini = (-1 * self.neural_net.len() as i32) + 1;
        let exp_output = exp_output.clone();

        //- Para cada capa:
        for (i, j) in (ini..0).zip((ini + 1)..1) {
            //- Si es la última capa:
            if i == ini {
                //- Calcular deltas última capa:
                //> Calcular error (e2medio = output layer - exp_output)
                let error = d_e2medio(&self.neural_net[(i * -1) as usize].output, &exp_output);

                //> Calcular deriv output layer
                let deriv = self.neural_net[(i * -1) as usize]
                    .activation_fuction
                    .derived_f(&self.neural_net[(i * -1) as usize].output);

                //> Calcular delta layer: error * deriv
                self.neural_net[(i * -1) as usize].set_deltas(&mult_mat(&error, &deriv));
            } else {
                //- Si no:
                //- Calcular deltas anteriores:
                //> Recuperación delta layer+1
                let delta = &self.neural_net[(i * -1) as usize + 1].deltas.clone();

                //> Recuperación weights layer+1
                let weights = &self.neural_net[(i * -1) as usize + 1].weights.clone();

                //> Transposición weights layer+1 -> w_t layer+1
                let weightst = weights.t();

                //> Recuperación output layer
                let output = &self.neural_net[(i * -1) as usize].output.clone();

                //> Calcular doo = deriv output layer
                let doo = self.neural_net[(i * -1) as usize]
                    .activation_fuction
                    .derived_f(&output);

                //> Calcular mult_mat = dot(delta layer+1, w_t layer+1)
                let mm = dot(&delta, &weightst);

                //> Calcular delta layer: mult_mat * doo
                self.neural_net[(i * -1) as usize].set_deltas(&mult_mat(&mm, &doo));
            }
            // sigue..

            // Descenso del Gradiente

            //- Actualización de bias:
            //> Calcular mean = mean(deltas layer)
            let mean = mean(&self.neural_net[(i * -1) as usize].deltas);

            //> Calcular mlr = mean * learning rate
            let mlr = mult_mat_float(&mean, learning_rate);

            //> Actualizar bias layer
            let b = &self.neural_net[(i * -1) as usize].bias;
            let nb = resta_mat(&b, &mlr);
            self.neural_net[(i * -1) as usize].set_bias(&nb);

            //- Actualización de weights:
            //> Recuperación output layer-1
            let out = &self.neural_net[(j * -1) as usize].output.clone();

            //> Transposición output layer-1 -> out_t layer-1
            let outt = out.t();

            //> Calcular do = dot(out_t layer-1, delta layer)
            let dooo = dot(&outt, &self.neural_net[(i * -1) as usize].deltas);

            //> Calcular dlr = dot * learning_rate
            let dlr = mult_mat_float(&dooo, learning_rate);

            //> Actualizar weights layer
            let w = &self.neural_net[(i * -1) as usize].weights;
            let nw = resta_mat(&w, &dlr);
            self.neural_net[(i * -1) as usize].set_weights(&nw);
        }
    }

    // Entrenamiento de la red neuronal
    pub fn train(&mut self, exp_output: Matriz, epochs: usize, learning_rate: f32) {
        print!("\nTraining...\n");
        
        for _i in 0..epochs {
            self.feed_forward();
            self.backpropagation(&exp_output, learning_rate);
        }
    }
}
