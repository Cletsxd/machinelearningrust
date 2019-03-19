mod functions;
mod math;
mod matriz;
mod neural_net;

use matriz::Matriz;
use neural_net::NeuralNet;

fn main() {
    //rustup doc --std
    //rustup doc --book
    // git checkout master
    // Sobre la red

    // PROBAR XOR //
    /*let layers = 3;
    let mut topo_nn = Vec::with_capacity(layers);
    topo_nn = [2, 3, 1].to_vec();*/

    // Sobre los datos de entrada de entrenamiento
    /*let rows_x = 4;
    let columns_x = 2;
    let mut vec_x = Vec::with_capacity(rows_x*columns_x);
    vec_x = [0.0,0.0, 1.0,0.0, 0.0,1.0, 1.0,1.0].to_vec();
    let X = Matriz::create_matriz(rows_x, columns_x, vec_x);*/

    // Sobre los datos de salida esperada
    /*let rows_y = 4;
    let columns_y = 1;
    let mut vec_y = Vec::with_capacity(rows_y*columns_y);
    vec_y = [0.0, 1.0, 1.0, 0.0].to_vec();
    let Y = Matriz::create_matriz(rows_y, columns_y, vec_y);*/

    // PROBAR UN SOLO PERCEPTRÓN //
    /*let layers = 2;
    let mut topo_nn = Vec::with_capacity(layers);
    topo_nn = [3, 1].to_vec();

    let rows_x = 4;
    let columns_x = 3;
    let mut vec_x = Vec::with_capacity(rows_x*columns_x);
    vec_x = [0.0,0.0,1.0, 1.0,1.0,1.0, 1.0,0.0,1.0, 0.0,1.0,1.0].to_vec();
    let X = Matriz::create_matriz(rows_x, columns_x, vec_x);

    let rows_y = 4;
    let columns_y = 1;
    let mut vec_y = Vec::with_capacity(rows_y*columns_y);
    vec_y = [0.0, 1.0, 1.0, 0.0].to_vec();
    let Y = Matriz::create_matriz(rows_y, columns_y, vec_y);

    // Creación de la ANN
    let mut ann = NeuralNet::create_neural_net(layers, X, topo_nn);*/

    // PROBAR COCHE AUTOMÁTICO //
    let layers = 3;
    let mut topo_nn = Vec::with_capacity(layers);
    topo_nn = [2, 3, 2].to_vec();

    let rows_x = 7;
    let columns_x = 2;
    let mut vec_x = Vec::with_capacity(rows_x * columns_x);
    vec_x = [
        0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.5, 1.0, 0.5, -1.0, 1.0, 1.0, 1.0, -1.0,
    ]
    .to_vec();
    let X = Matriz::create_matriz(rows_x, columns_x, vec_x);

    let rows_y = 7;
    let columns_y = 2;
    let mut vec_y = Vec::with_capacity(rows_y * columns_y);
    vec_y = [
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0, -1.0, 1.0, 1.0, 1.0, 0.0, -1.0, 0.0, -1.0,
    ]
    .to_vec();
    let Y = Matriz::create_matriz(rows_y, columns_y, vec_y);

    // Creación de la ANN
    let mut ann = NeuralNet::create_neural_net(layers, X, topo_nn);

    // Training
    ann.train(Y.clone(), 15000, 0.03);

    // mostrar ann
    print!("\nNeural Net after training\n");
    ann.show();

    // mostrar final output
    print!("Final Output\n");
    ann.show_final_output();

    print!("\nExpected Output\n");
    Y.show();

    // PROBAR UN SOLO PERCEPTRÓN
    // nueva entrada
    /*let r = 1;
    let c = 3;
    let mut nvec = Vec::with_capacity(r*c);
    nvec = [1.0,0.0,0.0].to_vec();
    let nx = Matriz::create_matriz(r, c, nvec);

    ann.feed_forward_wi(&nx);

    print!("\nNew input\n");
    nx.show();
    // mostrar final output
    print!("\nFinal Output\n");
    ann.show_final_output();*/
}
