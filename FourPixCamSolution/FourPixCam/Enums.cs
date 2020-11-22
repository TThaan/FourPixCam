namespace FourPixCam
{
    public enum ActivationType
    {
        Undefined, LeakyReLU, NullActivator,
        ReLU, Sigmoid, SoftMax, Tanh,
        None
    }
    public enum CostType
    {
        Undefined, SquaredMeanError
    }
    public enum WeightInitType
    {
        Undefined, Xavier,
        None
    }
    public enum Label
    {
        Undefined, AllBlack, AllWhite, LeftBlack, LeftWhite, SlashBlack, SlashWhite, TopBlack, TopWhite
    }
}