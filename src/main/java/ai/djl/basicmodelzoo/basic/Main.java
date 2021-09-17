package ai.djl.basicmodelzoo.basic;
import ai.djl.Application;
import ai.djl.Model;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.translate.TranslateException;

import java.awt.*;
import java.io.IOException;


public class Main {
    public static void main(String args[]){
        Application application = Application.CV.IMAGE_CLASSIFICATION;
        long inputSize = 28*28;
        long outputSize = 10;
        SequentialBlock block = new SequentialBlock();
        block.add(Blocks.batchFlattenBlock(inputSize));
        block.add(Linear.builder().setUnits(128).build());
        block.add(Activation::relu);
        block.add(Linear.builder().setUnits(64).build());
        block.add(Activation::relu);
        block.add(Linear.builder().setUnits(outputSize).build());
        int batchSize = 8;
        Mnist mnist = Mnist.builder().setSampling(batchSize, true).build();
        try {
            mnist.prepare(new ProgressBar());
            Model model = Model.newInstance("mlp");
            model.setBlock(new Mlp(28 * 28, 10, new int[] {128, 64}));
            DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                    .addEvaluator(new Accuracy())
                    .addTrainingListeners(TrainingListener.Defaults.logging());

            Trainer trainer = model.newTrainer(config);
            trainer.initialize(new Shape(1, 28*28));

            int epoch = 2;

            try {
                EasyTrain.fit(trainer, epoch, mnist, null);
            } catch (IOException e) {
                e.printStackTrace();
            } catch (TranslateException e) {
                e.printStackTrace();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }



    }
}
