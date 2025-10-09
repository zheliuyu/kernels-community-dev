{
  description = "Flake for rwkv kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder/rocm-per-source-arches";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genFlakeOutputs {
      path = ./.;
      rev = self.shortRev or self.dirtyShortRev or self.lastModifiedDate;
    };
}
