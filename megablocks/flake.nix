{
  description = "Flake for megablocks_moe kernel";

  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genFlakeOutputs {
      path = ./.;
      rev = self.shortRev or self.dirtyShortRev or self.lastModifiedDate;

      pythonCheckInputs = pkgs: with pkgs; [ 
        tqdm
        py-cpuinfo
        importlib-metadata
        torchmetrics
      ];
    };
}