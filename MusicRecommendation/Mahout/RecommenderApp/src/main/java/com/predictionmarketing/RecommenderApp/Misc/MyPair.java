package com.predictionmarketing.RecommenderApp.Misc;

public class MyPair {
	public long userID;
	public long itemID;
	public double value;
	public MyPair(long userID, long itemID, double value) {
		this.userID = userID;
		this.itemID = itemID;
		this.value = value;
	}
	public MyPair(MyPair p) {
		userID = p.userID;
		itemID = p.itemID;
		value = p.value;
	}
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(userID);
		sb.append(", ");
		sb.append(itemID);
		sb.append(", ");
		sb.append(value);
		return sb.toString();
	}
}