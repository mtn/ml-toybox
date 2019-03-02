use rand::{thread_rng, Rng};
use rand::distributions::{Normal, Distribution};

type LayerBiases = ndarray::Array2<f64>;
type LayerWeights = ndarray::Array2<f64>;

pub struct Network {
    layer_sizes: Vec<u32>,
    biases: Vec<LayerBiases>,
    weights: Vec<LayerWeights>,
}

impl Network {
    pub fn new(layer_sizes: Vec<u32>) -> Network {
        // Sample from N(0, 1)
        let normal = Normal::new(0., 1.);
        let mut rng = thread_rng();

        let biases = Self::init_biases(&layer_sizes, &mut rng, &normal);
        let weights = Self::init_weights(&layer_sizes, &mut rng, &normal);

        Network {
            layer_sizes,
            biases,
            weights,
        }
    }

    fn init_biases<R, T>(layer_sizes: &Vec<u32>, rng: R, distribution: &T) -> Vec<LayerBiases>
        where R: Rng, T: Distribution<f64>
    {
        vec![]
    }


    fn init_weights<R, T>(layer_sizes: &Vec<u32>, rng: &mut R, distribution: &T) -> Vec<LayerBiases>
        where R: Rng, T: Distribution<f64>
    {
        vec![]
    }
}
