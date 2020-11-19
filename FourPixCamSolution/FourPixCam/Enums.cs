namespace FourPixCam
{
    public enum ActivationType
    {
        Undefined, LeakyReLU, NullActivator,
        ReLU, Sigmoid, SoftMax
    }
    public enum Label
    {
        Undefined, AllBlack, AllWhite, LeftBlack, LeftWhite, SlashBlack, SlashWhite, TopBlack, TopWhite
    }
    public enum WeightInitialization
    {
        Undefined, Xavier
    }
}