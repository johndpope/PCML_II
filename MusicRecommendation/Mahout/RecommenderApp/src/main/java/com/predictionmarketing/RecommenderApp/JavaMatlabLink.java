package com.predictionmarketing.RecommenderApp;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.StringTokenizer;

import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.model.DataModel;

import com.predictionmarketing.RecommenderApp.Misc.MyPair;
import com.predictionmarketing.RecommenderApp.RecSystems.GlobalAverage;
import com.predictionmarketing.RecommenderApp.RecSystems.PerArtistAverage;
import com.predictionmarketing.RecommenderApp.RecSystems.RecSystemInterface;
import com.predictionmarketing.RecommenderApp.RecSystems.SimpleFactorization;
import com.predictionmarketing.RecommenderApp.RecSystems.SimpleNeighborhood;

public class JavaMatlabLink {

	private static String trainDataFile;
	private static String testDataFile;
	private static String outputFile;
	private static String algorithmName;
	private static HashMap<String, Object> parameters;
	
	public static void main(String[] args) throws IOException {
		
//		trainDataFile = "data/train.dat";
//		algorithmName = "kNN";
//		testDataFile = "data/test.dat";
//		outputFile = "data/output.txt";
		
		trainDataFile = args[0];
		testDataFile  = args[1];
 	    outputFile    = args[2];
		algorithmName = args[3];
		
		ArrayList<MyPair> trainPairs = loadData(trainDataFile);
		ArrayList<MyPair> testPairs = loadData(testDataFile);
		
		RecSystemInterface rec = null;
		
		if (algorithmName.equals("GlobalAverage")) {
			rec = new GlobalAverage();
		}
		if (algorithmName.equals("PerArtistAverage")) {
			rec = new PerArtistAverage();
		}
		if (algorithmName.equals("SimpleFactorization")) {
			rec = new SimpleFactorization();
		}
		if (algorithmName.equals("kNN")) {
			rec = new SimpleNeighborhood();
		}
		
		parameters = new HashMap<String, Object>();
		for(int idx = 4; idx < args.length; idx += 2) {
			parameters.put(args[idx], args[idx + 1]);
		}
		
//		parameters.put("N", "2");
		
		rec.setParameters(parameters);
		
		rec.Train(trainPairs);
		
		ArrayList<MyPair> ret = rec.Predict(testPairs);
		BufferedWriter out = new BufferedWriter(new FileWriter(new File(outputFile)));
		for(MyPair pair : ret) {
			out.write(pair.userID + "," + pair.itemID + "," + pair.value + "\n");
		}
		out.close();
	}
	
	
	
	private static ArrayList<MyPair> loadData(String filename) throws IOException {
		DataModel model = new FileDataModel(new File(filename));
		
		// we place the relationships in a hashmap
		HashMap<String, Long> hmRelationships = new HashMap<String, Long>();
		BufferedReader br = new BufferedReader(new FileReader(filename));
		while (true) {
			String line = br.readLine();
			if (line == null) {
				break;
			}
			StringTokenizer st = new StringTokenizer(line, ",");
			hmRelationships.put(st.nextToken()+" "+st.nextToken(), parseLong(st.nextToken()));
		}
		br.close();
		
		// we put all the relationships in array
		MyPair pairs[] = new MyPair[hmRelationships.size()];
		br = new BufferedReader(new FileReader(filename));
		int i = 0;
		while (true) {
			String line = br.readLine();
			if (line == null) {
				break;
			}
			StringTokenizer st = new StringTokenizer(line, ",");
			pairs[i] = new MyPair(Long.parseLong(st.nextToken()), Long.parseLong(st.nextToken()), parseLong(st.nextToken()));
			i++;
		}
		br.close();
		
		// we put all the relationships in array list
		ArrayList<MyPair> alPairs = new ArrayList<MyPair>();
		for (i=0;i<pairs.length;i++) {
			alPairs.add(pairs[i]);
		}
		return alPairs; 
	}
	
	static long parseLong(String s) {
		long num = 0;
		if (s.contains("e")) {
			double numD = Double.parseDouble(s.substring(0, s.indexOf('e')));
			int degree = Integer.parseInt(s.substring(s.indexOf('+')+1));
			for (int j = 0;j<degree;j++) {
				numD *= 10;
			}
			num = (long)numD;
		} else {
			num = Long.parseLong(s);
		}
		return num;
	}

}
