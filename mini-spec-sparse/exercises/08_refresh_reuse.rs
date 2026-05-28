#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerMode {
    Refresh,
    Reuse,
}

pub fn refresh_reuse_schedule(num_layers: usize, refresh_every: usize) -> Vec<LayerMode> {
    // TODO:
    // Layer 0 should refresh.
    // Every refresh_every layers should refresh.
    // Other layers reuse.
    // Treat refresh_every == 0 like refresh_every == 1.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedules_refreshes() {
        assert_eq!(
            refresh_reuse_schedule(6, 3),
            vec![
                LayerMode::Refresh,
                LayerMode::Reuse,
                LayerMode::Reuse,
                LayerMode::Refresh,
                LayerMode::Reuse,
                LayerMode::Reuse,
            ]
        );
    }

    #[test]
    fn refreshes_every_layer_when_period_is_one() {
        assert_eq!(
            refresh_reuse_schedule(3, 1),
            vec![LayerMode::Refresh, LayerMode::Refresh, LayerMode::Refresh]
        );
    }
}
